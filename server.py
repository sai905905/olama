from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ollama

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def ask_llama(prompt: str):
    res = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}])
    return {"response": res["message"]["content"]}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)