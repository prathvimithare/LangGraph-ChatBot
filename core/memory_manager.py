from langchain.memory import ConversationBufferWindowMemory

class MemoryManager:
    def __init__(self, k: int = 1):
        self.memory = ConversationBufferWindowMemory(k=k, return_messages=True, input_key="input", output_key="output")

    def load(self):
        return self.memory.load_memory_variables({}).get("history", [])

    def save(self, user_input: str, ai_response: str):
        self.memory.save_context({"input": user_input}, {"output": ai_response})
