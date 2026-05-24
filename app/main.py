from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat, exercice, rag

app = FastAPI(
    title="EduBénin IA API",
    description="Serveur IA — EduBénin v2",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    chat.router,
    prefix="/ia",
    tags=["Chat"]
)
app.include_router(
    exercice.router,
    prefix="/ia",
    tags=["Exercices"]
)
app.include_router(
    rag.router,
    prefix="/ia",
    tags=["RAG"]
)


@app.get("/")
def health_check():
    import os
    mode = os.getenv("MODEL_MODE", "openai")
    return {
        "status": "EduBénin IA opérationnel",
        "version": "2.0.0",
        "mode": mode,
        "modele": "gpt-5.5 (dev) | GLM-4.7 (prod)"
    }
