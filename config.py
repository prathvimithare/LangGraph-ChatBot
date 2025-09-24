import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")