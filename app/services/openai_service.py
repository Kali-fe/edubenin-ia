from openai import OpenAI
import json
import os

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", None)
    if base_url:
        return OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    return OpenAI(api_key=api_key)


def load_prompt(filename: str) -> str:
    import os as _os
    base_dir = _os.path.dirname(
        _os.path.dirname(
            _os.path.dirname(__file__)
        )
    )
    filepath = _os.path.join(
        base_dir, "prompts", filename
    )
    with open(
        filepath, "r", encoding="utf-8"
    ) as f:
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

    client = get_client()
    system_prompt = load_prompt(
        "base_system_prompt.txt"
    ).format(
        nom_eleve=nom_eleve,
        niveau=niveau,
        serie=serie or "Non spécifiée",
        matiere=matiere,
        score=score
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    messages += [
        {
            "role": m.role,
            "content": m.content
        }
        for m in historique
    ]
    messages.append(
        {"role": "user", "content": question}
    )

    response = client.chat.completions.create(
        model=os.getenv(
            "OPENAI_MODEL", "gpt-5.5"
        ),
        messages=messages,
        max_tokens=2000
    )
    return response.choices[0].message.content


def generer_exercice(
    matiere: str,
    niveau: str,
    serie: str,
    score: int,
    difficulte: int
) -> dict:
    client = get_client()
    prompt = load_prompt(
        "exercice_prompt.txt"
    ).format(
        matiere=matiere,
        niveau=niveau,
        serie=serie or "Non spécifiée",
        score=score,
        difficulte=difficulte
    )

    response = client.chat.completions.create(
        model=os.getenv(
            "OPENAI_MODEL", "gpt-5.5"
        ),
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "enonce": raw,
            "correction": "Non disponible",
            "difficulte": difficulte,
            "concepts_cles": [],
            "bareme": "",
            "criteres_correction": []
        }
