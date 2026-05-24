from fastapi import APIRouter
from app.models.schemas import (
    ChatRequest, ChatResponse
)
from app.services.model_factory import (
    get_model_service
)
import os

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse
)
def chat(req: ChatRequest):
    service = get_model_service()
    reponse = service.chat_avec_modele(
        question=req.question,
        nom_eleve=req.nom_eleve,
        niveau=req.niveau,
        serie=req.serie,
        matiere=req.matiere,
        score=req.score,
        historique=req.historique
    )
    mode = os.getenv("MODEL_MODE", "openai")
    modele = "gpt-5.5" if mode == "openai" \
        else "GLM-4.7 (Colab)"
    return ChatResponse(
        reponse=reponse,
        modele=modele
    )
