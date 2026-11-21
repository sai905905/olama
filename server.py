import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import ollama
from ddgs import DDGS

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


# --- ChromaDB + Smart Fallback RAG + Internet Search APIs ---

# Add a dedicated collection for global behavior rules
rules_collection = chroma_client.get_or_create_collection(name="system_rules")


@app.get("/rules")
def get_rules():
    return {"rules": rules_collection.get(include=["documents"])["documents"]}


@app.post("/rules")
def add_rule(data: dict):
    rule = data.get("rule", "").strip()
    if rule:
        rules_collection.upsert(ids=[f"rule_{int(time.time() * 1000)}"], documents=[rule])
    return get_rules()


def get_active_system_prompt(extra_instruction: str = "") -> str:
    saved_rules = rules_collection.get(include=["documents"])["documents"]
    rules_text = "\n".join(f"- {r}" for r in saved_rules) if saved_rules else "- Keep replies concise and realistic."
    return f"Follow these permanent user rules strictly:\n{rules_text}\n{extra_instruction}"

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
        return {
            "response": "I don't have that stored in ChromaDB yet.",
            "found_in_db": False,
            "sources": [],
        }

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
                    "Answer the question ONLY if the answer is explicitly supported by the Stored Context below. "
                    "If the Stored Context does NOT contain the answer, reply with the exact phrase: NOT_FOUND_IN_DB"
                ),
            },
            {
                "role": "user",
                "content": f"Stored Context:\n{context_block}\n\nQuestion: {question}",
            },
        ],
        options={"temperature": 0},
    )
    answer = res["message"]["content"].strip()
    if "NOT_FOUND_IN_DB" in answer:
        return {
            "response": "I couldn't find this in your local ChromaDB memory.",
            "found_in_db": False,
            "sources": retrieved_docs,
        }

    return {
        "response": answer,
        "found_in_db": True,
        "sources": retrieved_docs,
    }


@app.post("/web-rag")
def web_rag(data: dict):
    question = data.get("question", "").strip()
    snippets = []
    sources = []

    try:
        with DDGS() as ddgs:
            # Fetch top web results + latest news
            web_results = list(ddgs.text(question, max_results=3))
            news_results = list(ddgs.news(question, max_results=2))

        for r in web_results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            snippets.append(f"[Web] {title}: {body}")
            sources.append(f"{title} ({href})")

        for n in news_results:
            title = n.get("title", "")
            body = n.get("body", "")
            url = n.get("url", "")
            date = n.get("date", "")
            snippets.append(f"[News {date}] {title}: {body}")
            sources.append(f"[News] {title} ({url})")
    except Exception as e:
        return {"response": f"Internet search failed: {e}", "sources": []}

    if not snippets:
        return {"response": "No live internet results found for that query.", "sources": []}

    web_context = "\n".join(snippets)
    res = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. Summarize a clear, accurate answer to the user's question "
                    "using ONLY the live Internet & News search results provided below."
                ),
            },
            {
                "role": "user",
                "content": f"Live Search Results:\n{web_context}\n\nQuestion: {question}",
            },
        ],
        options={"temperature": 0.2},
    )
    return {
        "response": res["message"]["content"],
        "sources": sources,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)