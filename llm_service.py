from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class LLMService:
    def __init__(self, vector_store, ollama_host):
        self.llm = ChatOllama(
            model="llama3.2",
            base_url=ollama_host,
            temperature=0.7
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Build conversational chain with retrieval
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.vector_store.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )

    def get_response(self, query: str):
        try:
            response = self.chain({"question": query})
            answer = response["answer"]
            source_docs = response.get("source_documents", [])
            return answer, source_docs
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return "I encountered an error processing your request."
