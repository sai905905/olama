import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import ollama

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Persistent local ChromaDB folder
chroma_client = chromadb.PersistentClient(path="./test_chroma_db")
collection = chroma_client.get_or_create_collection(name="verify_rag")


def ask_llama(prompt: str):
    res = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}])
    return {"response": res["message"]["content"]}


def get_embedding(text: str):
    return ollama.embed(model="nomic-embed-text", input=text)["embeddings"][0]


# --- Existing APIs (Unchanged) ---
@app.get("/learn")
def learn():
    return {"about": "Local Llama 3.1:8b API running on FastAPI"}


@app.post("/converse")
def converse(data: dict):
    return ask_llama(data.get("message", ""))


@app.post("/ask")
def ask(data: dict):
    return ask_llama(data.get("question", ""))


@app.post("/topic")
def topic(data: dict):
    return ask_llama(f"Discuss this topic concisely: {data.get('topic', '')}")


# --- New ChromaDB + RAG APIs ---
@app.get("/memories")
def list_memories():
    data = collection.get(include=["documents"])
    items = [
        {"id": doc_id, "text": doc_text}
        for doc_id, doc_text in zip(data["ids"], data["documents"])
    ]
    return {"count": len(items), "items": items}


@app.post("/remember")
def remember_fact(data: dict):
    fact = data.get("fact", "").strip()
    if not fact:
        return {"status": "error", "message": "Empty fact"}
    doc_id = f"mem_{int(time.time() * 1000)}"
    collection.upsert(
        ids=[doc_id],
        embeddings=[get_embedding(fact)],
        documents=[fact],
    )
    return {"status": "saved", "id": doc_id, "fact": fact}


@app.post("/rag-ask")
def rag_ask(data: dict):
    question = data.get("question", "").strip()
    count = collection.count()
    if count == 0:
        return {"response": "ChromaDB is empty. Save a fact first!", "sources": []}

    q_emb = get_embedding(question)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(3, count),
    )
    retrieved_docs = results["documents"][0]
    context_block = "\n".join(f"- {d}" for d in retrieved_docs)

    res = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise personal & market assistant. "
                    "Answer the user's question clearly and concisely using ONLY the retrieved ChromaDB context below. "
                    "If the answer is not in the context, say you don't have that stored yet."
                ),
            },
            {
                "role": "user",
                "content": f"Stored Context:\n{context_block}\n\nQuestion: {question}",
            },
        ],
        options={"temperature": 0},
    )
    return {
        "response": res["message"]["content"],
        "sources": retrieved_docs,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)