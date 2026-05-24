import requests
import os


def get_colab_url():
    return os.getenv("COLAB_NGROK_URL", "")


def load_prompt(filename: str) -> str:
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    )
    import os as _os
    filepath = _os.path.join(
        base_dir, "prompts", filename
    )
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def chat_avec_modele(
    question: str,
    nom_eleve: str,
    niveau: str,
    serie: str,
    matiere: str,
    score: int,
    historique: list = None
) -> str:
    if historique is None:
        historique = []

    colab_url = get_colab_url()
    if not colab_url:
        return "Modèle Colab non disponible. Configure COLAB_NGROK_URL dans .env"

    response = requests.post(
        f"{colab_url}/ia/chat",
        json={
            "question": question,
            "nom_eleve": nom_eleve,
            "niveau": niveau,
            "serie": serie,
            "matiere": matiere,
            "score": score,
            "historique": [
                {"role": m.role, "content": m.content}
                for m in historique
            ]
        },
        timeout=60
    )
    data = response.json()
    return data.get("reponse", "Erreur modèle")


def generer_exercice(
    matiere: str,
    niveau: str,
    serie: str,
    score: int,
    difficulte: int
) -> dict:
    colab_url = get_colab_url()
    if not colab_url:
        return {
            "enonce": "Modèle Colab non disponible",
            "correction": "",
            "difficulte": difficulte,
            "concepts_cles": [],
            "bareme": "",
            "criteres_correction": []
        }

    response = requests.post(
        f"{colab_url}/ia/exercice",
        json={
            "matiere": matiere,
            "niveau": niveau,
            "serie": serie,
            "score": score,
            "difficulte": difficulte
        },
        timeout=60
    )
    return response.json()
