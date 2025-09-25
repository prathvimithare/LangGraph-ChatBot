from langchain_groq import ChatGroq

class LLMManager:
    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant", temperature: float = 0):
        self.llm = ChatGroq(model=model, groq_api_key=groq_api_key, temperature=temperature)

    def bind_tools(self, tools):
        self.llm = self.llm.bind_tools(tools)
        return self.llm
