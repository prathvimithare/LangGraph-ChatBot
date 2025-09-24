from config import Config
from opensearch import OpenSearchClient
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_service import LLMService

index_name = "cocobp"
bp_client = OpenSearchClient(host=Config.OPENSEARCH_URL, index_name=index_name)

all_docs = bp_client.get_all_documents()
print(f"Total documents fetched: {len(all_docs)}")

new_docs = bp_client.get_new_documents()
print(f"Total documents fetched: {len(new_docs)}")

all_docs = all_docs + new_docs

processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

keys_to_use = [
                "title", "challengeLongDescription", 
                "solutionLongDescription", "benefitsLongDescription"
            ]
texts = processor.extract_text(all_docs, keys_to_use)

chunks = processor.split_documents(texts)

print(f"Total chunks created: {len(chunks)}")

vector_client = VectorStore(collection_name="cocobp_chunks", persist_directory="db/chroma", ollama_host=Config.OLLAMA_HOST)
vector_client.add_documents(chunks)
print("Added chunks to vector store")

llm_chat = LLMService(vector_client, Config.OLLAMA_HOST)

query = "What are the main challenges in AI adoption?"
answer, source_docs = llm_chat.get_response(query)

print("User:", query)
print("AI:", answer)

print("\nReferenced Documents:")
for i, source_doc in enumerate(source_docs, 1):
    print(f"{i}. {source_doc.metadata.get('title', 'Unknown source')}")
    print(f"   Content: {source_doc.page_content[:200]}...")