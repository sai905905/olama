import chromadb
import ollama

client = chromadb.PersistentClient(path="./test_chroma_db")
collection = client.get_or_create_collection(name="verify_rag")

docs = [
    "Project Falcon uses Upstox Sandbox port 8090 and caps trade risk at 1.5% per order.",
    "Laurus Labs Q3 internal target price in our private watchlist is ₹640 with a stop-loss at ₹585.",
]

for i, doc in enumerate(docs):
    emb = ollama.embed(model="nomic-embed-text", input=doc)["embeddings"][0]
    collection.upsert(ids=[f"doc_{i}"], embeddings=[emb], documents=[doc])

print("ChromaDB count:", collection.count())

question = "What is our private watchlist target and stop-loss for Laurus Labs?"
q_emb = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]
retrieved = collection.query(query_embeddings=[q_emb], n_results=1)["documents"][0][0]
print("Retrieved:", retrieved)

res = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": f"Answer using ONLY this context:\n{retrieved}\n\nQuestion: {question}"}],
    options={"temperature": 0},
)
print("RAG Answer:", res["message"]["content"])