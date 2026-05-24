from pydantic import BaseModel, Field
from typing import List, Optional


class MessageHistory(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    nom_eleve: str
    niveau: str
    serie: Optional[str] = None
    matiere: str
    score: int = Field(default=50, ge=0, le=100)
    historique: List[MessageHistory] = Field(
        default_factory=list
    )


class ChatResponse(BaseModel):
    reponse: str
    modele: str


class ExerciceRequest(BaseModel):
    matiere: str
    niveau: str
    serie: Optional[str] = None
    score: int = Field(default=50, ge=0, le=100)
    difficulte: int = Field(
        default=5, ge=1, le=10
    )


class ExerciceResponse(BaseModel):
    enonce: str
    correction: str
    difficulte: int
    concepts_cles: List[str]
    bareme: Optional[str] = None
    criteres_correction: List[str] = Field(
        default_factory=list
    )


class RAGIntegrerRequest(BaseModel):
    contenu: str
    matiere: str
    classe: str
    serie: Optional[str] = None
    source: str
    source_id: Optional[str] = None


class RAGIntegrerResponse(BaseModel):
    chunks_crees: int
    message: str
