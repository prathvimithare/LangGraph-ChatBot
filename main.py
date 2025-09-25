from core.llm_manager import LLMManager
from core.memory_manager import MemoryManager
from core.agent_graph import create_graph
from core.prompts import SYSTEM_PROMPT
from services.vector_service import VectorService
from langchain_core.messages import HumanMessage
from config import Config
import tools.papers_tool as pt
import tools.bp_tool as bt
from tools.papers_tool import papers_tool
from tools.bp_tool import bp_tool

def main():
    # Initialize LLM and Memory
    llm_manager = LLMManager(Config.GROQ_API_KEY)
    memory = MemoryManager()

    papers_service = VectorService(
    index_name="papers",
    collection_name="papers_chunks",
    text_keys=["title", "abstr"],
    metadata_keys=["title", "id"]
    )

    bp_service = VectorService(
        index_name="cocobp",
        collection_name="cocobp_chunks",
        text_keys=["title", "challengeLongDescription", "solutionLongDescription", "benefitsLongDescription"],
        metadata_keys=["title", "csId"]
    )

    papers_service.index_documents()               
    bp_service.index_documents()

    top_docs = 5
    conf_threshold = 0.8
    papers_retriever = papers_service.build_retriever(top_docs, conf_threshold)  
    bp_retriever = bp_service.build_retriever(top_docs, conf_threshold)


    # Assign retrievers to tool modules
    pt.papers_retriever = papers_retriever
    bt.bp_retriever = bp_retriever

    tools = [papers_tool, bp_tool]
    tools_dict = {t.name: t for t in tools}

    # Bind tools to LLM
    llm = llm_manager.bind_tools(tools)

    # Create RAG agent graph
    rag_agent = create_graph(llm, tools_dict, SYSTEM_PROMPT)

    print("\n=== RAG AGENT ===")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        history = memory.load()
        messages = history + [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        ai_response = result["messages"][-1].content

        print("\n=== ANSWER ===")
        print(ai_response)

        memory.save(user_input, ai_response)


if __name__ == "__main__":
    main()
