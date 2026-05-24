import json
import numpy as np
from openai import OpenAI
import os


def get_client():
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )


def generer_embedding(texte: str) -> list:
    client = get_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texte[:8000]
    )
    return response.data[0].embedding


def similarite_cosinus(v1, v2) -> float:
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(
        np.dot(v1, v2) /
        (np.linalg.norm(v1) * np.linalg.norm(v2))
    )


def chunk_texte(
    texte: str,
    taille: int = 500
) -> list:
    mots = texte.split()
    return [
        " ".join(mots[i:i+taille])
        for i in range(0, len(mots), taille)
    ]


def rechercher_contexte_rag(
    question: str,
    documents_rag: list,
    top_k: int = 3
) -> str:
    if not documents_rag:
        return ""

    question_emb = generer_embedding(question)
    scores = []

    for doc in documents_rag:
        if doc.get("embedding"):
            emb = json.loads(doc["embedding"])
            score = similarite_cosinus(
                question_emb, emb
            )
            scores.append(
                (score, doc["contenu"])
            )

    scores.sort(reverse=True)
    top_docs = [doc for _, doc in scores[:top_k]]

    if not top_docs:
        return ""

    return "\n\n".join([
        f"Exemple réel béninois:\n{doc}"
        for doc in top_docs
    ])
