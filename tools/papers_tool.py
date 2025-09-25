from langchain_core.tools import tool

# Retriever will be set externally in main.py
papers_retriever = None

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
    if papers_retriever is None:
        return "Retriever not set."
    
    docs, scores = papers_retriever.invoke(query)
    # print(f'\nNo. of docs: {len(docs)} and scores: {scores}. \nThe retrieved documents are: {docs}\n')
    if not docs:
        return "No relevant publications found."
    
    results = [
        f"[Source: {d.metadata.get('title','Unknown')} | ID: {d.metadata.get('id','Unknown')}]\nContent: {d.page_content}"
        for d in docs
    ]
    return "\n\n".join(results)
