from fastapi import FastAPI
from src.api.routes import chat, documents, health
from src.api.routes import chat, documents

app = FastAPI(title="Legal RAG Assistant", version="0.1.0")

app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(health.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
