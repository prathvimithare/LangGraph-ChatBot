from langchain_core.tools import tool

# Retriever will be set externally in main.py
bp_retriever = None

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
    if bp_retriever is None:
        return "Retriever not set."
    
    docs, scores = bp_retriever.invoke(query)
    # print(f'\nNo. of docs: {len(docs)} and scores: {scores}. \nThe retrieved documents are: {docs}\n')
    if not docs:
        return "No relevant Praxisbeispiele found."
    
    results = [
        f"[Source: {d.metadata.get('title','Unknown')} | csID: {d.metadata.get('csId','Unknown')}]\nContent: {d.page_content}"
        for d in docs
    ]
    return "\n\n".join(results)
