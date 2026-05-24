import openai
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MATIERES_CLASSES = {
    "PCT": {
        "classes": ["3eme", "Terminale"],
        "series": [None, "C", "D"]
    },
    "SVT": {
        "classes": ["3eme", "Terminale"],
        "series": [None, "A", "B", "C", "D"]
    },
    "Mathematiques": {
        "classes": ["3eme", "Terminale"],
        "series": [None, "A", "B", "C", "D"]
    },
    "Anglais": {
        "classes": ["3eme", "Terminale"],
        "series": [None, "A", "B", "C", "D"]
    },
    "Histoire-Geographie": {
        "classes": ["3eme", "Terminale"],
        "series": [None, "A", "B", "C", "D"]
    }
}

TYPES = ["explication", "exercice", "dialogue"]


def charger_coefficients() -> dict:
    fichier = Path("dataset/coefficients.json")
    if not fichier.exists():
        raise FileNotFoundError(
            "dataset/coefficients.json introuvable"
        )
    with open(fichier, encoding="utf-8") as f:
        return json.load(f)


def normaliser_coefficient(valeur) -> int:
    if valeur is None:
        return 1
    try:
        coeff = int(valeur)
    except (TypeError, ValueError):
        return 1
    return max(coeff, 1)


def charger_contexte() -> str:
    processed_dir = Path("dataset/processed")
    fichiers_legacy = [
        processed_dir / "documents_analyses.json"
    ]
    fichiers_sep = list(
        sorted((processed_dir / "cours").glob("*.json"))
    ) + list(
        sorted((processed_dir / "epreuves").glob("*.json"))
    )

    docs = []
    for fichier in fichiers_sep + fichiers_legacy:
        if not fichier.exists():
            continue

        with open(fichier, encoding="utf-8") as f:
            contenu = json.load(f)

        if isinstance(contenu, list):
            docs.extend(contenu)
        else:
            docs.append(contenu)

        if len(docs) >= 3:
            break

    if docs:
        return json.dumps(
            docs[:3], ensure_ascii=False
        )[:3000]
    return "Aucun document de référence chargé."


def generer_exemple(
    client,
    matiere: str,
    classe: str,
    serie: str,
    type_ex: str,
    contexte: str,
    compteur: int
) -> dict:
    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {
                "role": "system",
                "content": f"""Tu es EduBot,

RÈGLES ABSOLUES :
- Français uniquement
- Programme officiel béninois (MENB)
- Contexte béninois concret
- Ne jamais donner la réponse directement
- Guider par questions progressives
- Encourager l'élève

RÉFÉRENCE PROGRAMME :
{contexte}

Génère UN exemple de type {type_ex}
{{
  "instruction": "question ou consigne",
  "output": "réponse complète du tuteur",
  "bareme": "si applicable sinon null",
  "criteres_correction": ["critère 1"]
}}"""
            },
            {
                "role": "user",
                "content": f"Matière: {matiere}\n"
                          f"Classe: {classe}\n"
                          f"Série: {serie or 'N/A'}\n"
                          f"Type: {type_ex}\n"
                          f"Exemple #{compteur}"
            }
        ],
        max_tokens=1500
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception:
        return {
            "instruction": f"Question {compteur}",
            "output": raw,
            "bareme": None,
            "criteres_correction": []
        }


def generer_dataset():
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    coefficients = charger_coefficients()
    contexte = charger_contexte()
    dataset = []
    compteur = 0

    for classe, matieres in coefficients.get("classes", {}).items():
        for matiere, coeff in matieres.items():
            coefficient = normaliser_coefficient(coeff)
            nb_exemples = coefficient * 5
            config = MATIERES_CLASSES.get(matiere, {})
            series = config.get("series", [None])

            for serie in series:
                for type_ex in TYPES:
                    for _ in range(nb_exemples):
                        compteur += 1
                        try:
                            ex = generer_exemple(
                                client, matiere,
                                classe, serie,
                                type_ex, contexte,
                                compteur
                            )
                            matiere_id = matiere \
                                .lower() \
                                .replace("-", "") \
                                .replace(" ", "")[:4]
                            serie_id = serie or "all"
                            entry = {
                                "id": f"edu-{matiere_id}-{classe}-{serie_id}-{compteur:04d}",
                                "type": type_ex,
                                "matiere": matiere,
                                "classe": classe,
                                "serie": serie,
                                "coefficient": coefficient,
                                "source": "genere_gpt5.5",
                                "valide": False,
                                **ex
                            }
                            dataset.append(entry)
                            print(
                                f"✅ {compteur} — "
                                f"{classe} {matiere} "
                                f"{serie or ''} {type_ex}"
                            )
                        except Exception as e:
                            print(f"❌ Erreur #{compteur}: {e}")

    output = Path(
        "dataset/edubenin_dataset_v2.json"
    )
    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            dataset, f,
            ensure_ascii=False, indent=2
        )
    print(f"\n✅ {len(dataset)} exemples générés")
    print(f"📁 Fichier: {output}")


if __name__ == "__main__":
    generer_dataset()
