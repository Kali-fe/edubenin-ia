import requests
import os


def get_colab_url():
    return os.getenv("COLAB_NGROK_URL", "").strip()


def load_prompt(filename: str) -> str:
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    )
    filepath = os.path.join(base_dir, "prompts", filename)
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

    # Construire le message complet pour EduBot
    message = f"[Élève: {nom_eleve} | Niveau: {niveau} | Matière: {matiere} | Score: {score}/100]\n{question}"

    try:
        response = requests.post(
            f"{colab_url}/ia/chat",
            json={
                "message": message,
                "classe": niveau,
                "matiere": matiere
            },
            timeout=120,
            headers={"ngrok-skip-browser-warning": "true"}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "Erreur modèle")
    except requests.exceptions.Timeout:
        return "EduBot met trop de temps à répondre. Réessaie dans quelques secondes."
    except Exception as e:
        return f"Erreur connexion EduBot : {str(e)}"


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

    try:
        response = requests.post(
            f"{colab_url}/ia/exercice",
            json={
                "matiere": matiere,
                "niveau": niveau,
                "serie": serie,
                "score": score,
                "difficulte": difficulte
            },
            timeout=120,
            headers={"ngrok-skip-browser-warning": "true"}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "enonce": f"Erreur : {str(e)}",
            "correction": "",
            "difficulte": difficulte,
            "concepts_cles": [],
            "bareme": "",
            "criteres_correction": []
        }