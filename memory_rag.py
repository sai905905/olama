import chromadb
import ollama

client = chromadb.PersistentClient(path="./test_chroma_db")
collection = client.get_or_create_collection(name="verify_rag")

# 1. Store personal fact permanently in ChromaDB
fact = "Sathvika is my wife."
emb = ollama.embed(model="nomic-embed-text", input=fact)["embeddings"][0]
collection.upsert(ids=["personal_1"], embeddings=[emb], documents=[fact])

# 2. Ask via RAG (searches ChromaDB first, then forces Llama to use it)
question = "Who is Sathvika?"
q_emb = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]
retrieved = collection.query(query_embeddings=[q_emb], n_results=1)["documents"][0][0]

res = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {
            "role": "system",
            "content": "Answer strictly using ONLY the provided personal context. Do not use outside trivia.",
        },
        {
            "role": "user",
            "content": f"Context: {retrieved}\n\nQuestion: {question}",
        },
    ],
    options={"temperature": 0},
)
print("RAG Answer:", res["message"]["content"])