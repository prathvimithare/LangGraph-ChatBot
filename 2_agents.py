# import os
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

# --- LOAD ENV ---
GROQ_API_KEY = Config.GROQ_API_KEY
ollama_host = Config.OLLAMA_HOST

# --- INITIALIZE LLM ---
llm = ChatGroq(
    model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY, temperature=0
)

# --- MEMORY ---
memory = ConversationBufferWindowMemory(k=3, return_messages=True, input_key="input", output_key="output")

# --- DOCUMENT PROCESSOR ---
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

# ======================
# --- PAPERS TOOL SETUP ---
# ======================
papers_index_name = "papers"
papers_client = OpenSearchClient(host=Config.OPENSEARCH_URL, index_name=papers_index_name)
papers_docs = papers_client.get_all_documents()
print(f"Total paper documents fetched: {len(papers_docs)}")

papers_texts = processor.extract_text(
    papers_docs,
    keys=["title", "abstr"],
    metadata_keys=["title", "id"]
)
papers_chunks = processor.split_documents(papers_texts)

papers_vector_client = VectorStore(
    collection_name="papers_chunks",
    persist_directory="ChromaDB",
    ollama_host=ollama_host
)

# papers_vector_client.add_documents(papers_chunks)
# print(f"Added {len(papers_chunks)} chunks to vector store")

papers_retriever = papers_vector_client.vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

@tool
def papers_tool(query: str) -> str:
    """
    This tool is designed to search and return relevant scientific publications 
    from the knowledge base. 

    It should **only be used when the user asks for information about research papers, 
    scientific studies, or authoritative references** on a specific topic. 

    Each retrieved publication includes:
    - The **title** and **abstract** to provide a clear understanding of the research content
    - Metadata for proper citation, including the publication **title** and **ID**

    Do **not** use this tool for casual conversation or questions unrelated to publications. 
    It is intended solely for queries seeking credible references and scientific content.
    """
    docs = papers_retriever.invoke(query)
    # print(f'\nThe retrieved documents are: {docs}\n')
    if not docs:
        return "No relevant publications found."
    results = [
        f"[Source: {d.metadata.get('title','Unknown')} | ID: {d.metadata.get('id','Unknown')}]\nContent: {d.page_content}"
        for d in docs
    ]
    return "\n\n".join(results)

# ===========================
# --- BUSINESS PRACTICE TOOL ---
# ===========================
bp_index_name = "cocobp"
bp_client = OpenSearchClient(host=Config.OPENSEARCH_URL, index_name=bp_index_name)
bp_docs = bp_client.get_all_documents()
print(f"Total BP documents fetched: {len(bp_docs)}")

bp_texts = processor.extract_text(
    bp_docs,
    keys=["title", "challengeLongDescription", "solutionLongDescription", "benefitsLongDescription"],
    metadata_keys=["title", "csId"]
)
bp_chunks = processor.split_documents(bp_texts)

bp_vector_client = VectorStore(
    collection_name="cocobp_chunks",
    persist_directory="ChromaDB",
    ollama_host=ollama_host
)

# bp_vector_client.add_documents(bp_chunks)
# print(f"Added {len(bp_chunks)} chunks to vector store")

bp_retriever = bp_vector_client.vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

@tool
def bp_tool(query: str) -> str:
    """
    This tool is designed specifically to search and return relevant Praxisbeispiele 
    (practical examples) from research, application, and current trends. 

    It should **only be used when the user asks about practical examples, case studies, 
    or related applications in the context of business challenges 
    ('Betriebliche Herausforderungen') or work science topics 
    ('Themen der Arbeitswissenschaft')**. 

    For each result, the tool provides:
    - The text of the example (challenge, solution, benefits)
    - Metadata for proper citation, including the title and csID

    Do **not** use this tool for general conversation or unrelated questions. 
    It is intended only for queries seeking concrete Praxisbeispiele.
    """
    docs = bp_retriever.invoke(query)
    # print(f'\nThe retrieved documents are: {docs}\n')
    if not docs:
        return "No relevant Praxisbeispiele found."
    results = [
        f"[Source: {d.metadata.get('title','Unknown')} | csID: {d.metadata.get('csId','Unknown')}]\nContent: {d.page_content}"
        for d in docs
    ]
    return "\n\n".join(results)

# --- BIND TOOLS ---
tools = [papers_tool, bp_tool]
llm = llm.bind_tools(tools)
tools_dict = {t.name: t for t in tools}

# --- AGENT STATE ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

# --- SYSTEM PROMPT ---
system_prompt = """
You are an intelligent AI assistant specialized in answering questions using two types of knowledge:

1. **Scientific publications** stored in the knowledge base.  
   Each publication includes a **title** and an **abstract**, which provide insight into the research topic and content.

2. **Praxisbeispiele** (practical examples) from research, application, and current trends.  
   These examples provide concrete insights, inspiration, and a valuable resource for addressing interdisciplinary and cross-company challenges.  
   Praxisbeispiele can be explored through the perspectives of **Betriebliche Herausforderungen** (business challenges) and **Themen der Arbeitswissenschaft** (work science topics).

When generating responses:

- Always prioritize the **most recent user message**. Treat it as the primary source of intent.  
- Use past history only if it directly clarifies or relates to the new query.
- If the new message introduces a new topic, do not let older history override it.  
- Base your answers solely on information retrieved from the knowledge base or Praxisbeispiele.  
- If you have used tools to answer, **never forget to include the metadata citation at the end of the answer**.  
- If the knowledge base does not contain sufficient information, clearly state this instead of making up an answer.

**Tool usage guidance:**

- Carefully analyze the user's query before calling any tool.  
- Only call the **publications tool** when the user explicitly asks about research papers, scientific studies, or authoritative references.  
- Only call the **Praxisbeispiele tool** when the user is asking about practical examples, case studies, or work science/business challenge scenarios.  
- Do not call any tool for general conversation or unrelated questions.  
- You must only call tools that are explicitly available in the provided list.  
- Do not invent or call tools outside the available tools.
"""

# system_prompt = """
# You are a knowledgeable AI assistant specialized in answering questions based on the documents stored in your knowledge base. 
# Use the available retriever tools **only when the user explicitly mentions "publications" or "Praxisbeispiele"** in their query. 
# Otherwise, you may engage in normal conversation without invoking any tools. 
# You can make multiple calls to the tools if needed to gather context when appropriate. 

# When responding:
# - Always prioritize the **most recent user message**. 
# - Base your answers only on information retrieved from the knowledge base when tools are called. 
# - Cite the specific parts of the documents you use, including metadata like title, ID, or page number. 
# - If the knowledge base does not contain sufficient information, state this clearly instead of making up an answer.
# - Do not use any tools or external information not explicitly provided.
# """



# --- LLM CALL ---
def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    message = llm.invoke(messages)
    return {'messages': [message]}

# --- TOOL EXECUTION ---
def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query','No query provided')}")
        if t['name'] not in tools_dict:
            print(f"\nTool {t['name']} does not exist.")
            result = "Incorrect tool name. Please select a valid tool."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query',''))
            print(f"Result length: {len(str(result))}")
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools execution complete.")
    return {'messages': results}

# --- STATE GRAPH ---
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()

# --- RUNNING AGENT ---
def running_agent():
    print("\n=== RAG AGENT===")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        # --- Load conversation history ---
        history = memory.load_memory_variables({})["history"]
        messages = history + [HumanMessage(content=user_input)]

        # --- Invoke agent ---
        result = rag_agent.invoke({"messages": messages})
        ai_response = result['messages'][-1].content

        # --- Display answer ---
        print("\n=== ANSWER ===")
        print(ai_response)

        # --- Save to memory ---
        memory.save_context({"input": user_input}, {"output": ai_response})

running_agent()
