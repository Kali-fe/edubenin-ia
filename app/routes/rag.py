from fastapi import APIRouter
from app.models.schemas import (
    RAGIntegrerRequest, RAGIntegrerResponse
)
from app.services.rag_service import (
    chunk_texte, generer_embedding
)

router = APIRouter()


@router.post(
    "/rag/integrer",
    response_model=RAGIntegrerResponse
)
def integrer_rag(req: RAGIntegrerRequest):
    chunks = chunk_texte(req.contenu)
    nb_integres = 0

    for chunk in chunks:
        try:
            generer_embedding(chunk)
            nb_integres += 1
        except Exception as e:
            print(f"Erreur embedding chunk: {e}")

    return RAGIntegrerResponse(
        chunks_crees=nb_integres,
        message=f"{nb_integres} chunks intégrés"
    )
