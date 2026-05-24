from fastapi import APIRouter
from app.models.schemas import (
    ExerciceRequest, ExerciceResponse
)
from app.services.model_factory import (
    get_model_service
)

router = APIRouter()


@router.post(
    "/exercice",
    response_model=ExerciceResponse
)
def exercice(req: ExerciceRequest):
    service = get_model_service()
    result = service.generer_exercice(
        matiere=req.matiere,
        niveau=req.niveau,
        serie=req.serie,
        score=req.score,
        difficulte=req.difficulte
    )
    return ExerciceResponse(
        enonce=result.get("enonce", ""),
        correction=result.get("correction", ""),
        difficulte=result.get(
            "difficulte", req.difficulte
        ),
        concepts_cles=result.get(
            "concepts_cles", []
        ),
        bareme=result.get("bareme", ""),
        criteres_correction=result.get(
            "criteres_correction", []
        )
    )
