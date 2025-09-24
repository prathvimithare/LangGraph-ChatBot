from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.memory import ConversationBufferWindowMemory

from config import Config
from opensearch import OpenSearchClient
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_service import LLMService

load_dotenv()

GROQ_API_KEY = Config.GROQ_API_KEY
ollama_host = Config.OLLAMA_HOST

llm = ChatGroq(
    model = "llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY, temperature = 0) # I want to minimize hallucination - temperature = 0 makes the model output more deterministic 

memory = ConversationBufferWindowMemory(k=3, return_messages=True, input_key="input", output_key="output")

index_name = "cocobp"
bp_client = OpenSearchClient(host=Config.OPENSEARCH_URL, index_name=index_name)

all_docs = bp_client.get_all_documents()
print(f"Total documents fetched: {len(all_docs)}")

processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

keys_to_use = [
                "title", "challengeLongDescription", 
                "solutionLongDescription", "benefitsLongDescription"
            ]
metadata_keys = [
    "title", "csId"
]
texts = processor.extract_text(all_docs, keys_to_use, metadata_keys)

chunks = processor.split_documents(texts)

vector_client = VectorStore(collection_name="cocobp_chunks", persist_directory="ChromaDB", ollama_host=Config.OLLAMA_HOST)
# vector_client.add_documents(chunks)
print("Added chunks to vector store")

retriever = vector_client.vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns relevant Praxisbeispiele (practical examples) 
    from research, application, and current trends. 
    It helps users explore content through the perspectives of 
    business challenges ("Betriebliche Herausforderungen") and 
    work science topics ("Themen der Arbeitswissenschaft"). 
    The tool provides both the example text and its metadata (e.g., title, ID) 
    for citation in answers.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information."
    
    results = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "Unknown Title")
        doc_id = doc.metadata.get("csId", "Unknown ID")
        results.append(
        f"[Source: {title} | csID: {doc_id}]\nContent: {doc.page_content}"
    )
    
    return "\n\n".join(results)

# q = "Die bislang häufig manuell und intuitiv durchgeführte Personaleinsatzplanung wird insb. in KMU zunehmend zur komplexen Planungsaufgabe."
tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant specialized in answering questions about "Praxisbeispiele" (practical examples) from research, application, and current trends. 
These examples provide concrete insights, serve as inspiration, and act as a valuable resource for addressing interdisciplinary and cross-company challenges.

The Praxisbeispiele can be explored through the perspectives of "Betriebliche Herausforderungen" (business challenges) and "Themen der Arbeitswissenschaft" (work science topics).

When generating responses:
- Always prioritize the **most recent user message**. Treat it as the primary source of intent.  
- Use previous conversation history only if it adds clarity, context, or continuity.  
- If the new message introduces a new topic, do not let older history override it.  
- Always base your answers on information retrieved from the knowledge base.  
- If relevant, cite the metadata (such as title or csID) of the source so users know where the information comes from.  
- If the knowledge base does not contain sufficient information, state this clearly instead of making up an answer.
- You must only call tools that are explicitly available in the provided list. 
- If a tool is not listed, do not attempt to use it. 
- Never invent or call tools or anything else outside the available tools.
"""


tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        # --- Load conversation history (last 3 turns) ---
        history = memory.load_memory_variables({})["history"]

        # Convert memory history into messages + new input
        messages = history + [HumanMessage(content=user_input)]

        # print(f'\n\nmessages is: {messages}\n\n')

        # messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        ai_response = result['messages'][-1].content
        print("\n=== ANSWER ===")
        print(ai_response)

        # --- Save user input + AI response into memory (together) ---
        memory.save_context(
            {"input": user_input},  # input dict
            {"output": ai_response} # output dict
        )


running_agent()




