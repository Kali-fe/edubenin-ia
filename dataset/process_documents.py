import fitz
import openai
import json
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def extraire_pdf(chemin: str) -> str:
    doc = fitz.open(chemin)
    return "\n".join(
        page.get_text() for page in doc
    )


def analyser_document(
    texte: str,
    nom_fichier: str
) -> dict:
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {
                "role": "system",
                "content": """Tu es expert du 
Analyse ce document et extrais en JSON :
{
  "type": "cours|epreuve|corrige|programme",
  "matiere": "...",
  "classe": "3eme|Terminale",
  "serie": "A|B|C|D|null",
  "chapitres": ["..."],
  "notions_cles": ["..."],
  "exercices": [
    {
      "enonce": "...",
      "corrige": "...",
      "bareme": "...",
      "criteres": ["..."]
    }
  ],
  "contenu_structure": "résumé structuré"
}
Réponds UNIQUEMENT en JSON valide."""
            },
            {
                "role": "user",
                "content": f"Fichier: {nom_fichier}\n\n{texte[:8000]}"
            }
        ],
        max_tokens=4000
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception:
        return {
            "type": "inconnu",
            "fichier": nom_fichier,
            "contenu_brut": raw
        }


def charger_coefficients() -> dict:
    fichier = Path("dataset/coefficients.json")
    if not fichier.exists():
        raise FileNotFoundError(
            "dataset/coefficients.json introuvable"
        )
    with open(fichier, encoding="utf-8") as f:
        return json.load(f)


def trouver_coefficient(
    coefficients: dict,
    classe: str,
    matiere: str
) -> Optional[int]:
    return coefficients.get("classes", {}) \
        .get(classe, {}) \
        .get(matiere)


def lister_dossiers_classes(
    base_dir: Path,
    exclus: Optional[set[str]] = None
) -> list[Path]:
    if not base_dir.exists():
        return []

    exclus = exclus or set()
    return [
        d for d in base_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and d.name not in exclus
    ]


def lire_document(fichier: Path) -> str:
    if fichier.suffix.lower() == ".pdf":
        return extraire_pdf(str(fichier))
    return fichier.read_text(encoding="utf-8")


def traiter_type_documents(
    input_dir: Path,
    output_dir: Path,
    type_document: str,
    coefficients: dict,
    exclus: Optional[set[str]] = None
) -> list[dict]:
    dossiers_classe = lister_dossiers_classes(
        input_dir, exclus=exclus
    )

    if not input_dir.exists():
        print(f"ℹ️ Dossier absent: {input_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"📂 {len(dossiers_classe)} classes trouvees dans "
        f"{input_dir}"
    )

    resultats = []
    for dossier_classe in dossiers_classe:
        classe = dossier_classe.name
        for dossier_matiere in lister_dossiers_classes(
            dossier_classe
        ):
            matiere = dossier_matiere.name
            coefficient = trouver_coefficient(
                coefficients, classe, matiere
            )
            print(
                f"⏳ Traitement {type_document}: "
                f"{classe}/{matiere} "
                f"(coefficient={coefficient})"
            )
            fichiers = [
                f for f in dossier_matiere.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]

            if not fichiers:
                continue

            analyses = []
            for fichier in fichiers:
                try:
                    texte = lire_document(fichier)
                    analyse = analyser_document(
                        texte, fichier.name
                    )
                    analyse["type"] = type_document
                    analyse["fichier_source"] = fichier.name
                    analyse["classe"] = classe
                    analyse["matiere"] = matiere
                    analyse["coefficient"] = coefficient
                    analyses.append(analyse)
                    resultats.append(analyse)
                    print(f"✅ {fichier.name} traite")
                except Exception as e:
                    print(f"❌ Erreur {fichier.name}: {e}")

            if analyses:
                output = output_dir / (
                    f"{classe}_{matiere}_{type_document}.json"
                )
                with open(output, "w", encoding="utf-8") as f:
                    json.dump(
                        analyses, f,
                        ensure_ascii=False, indent=2
                    )
                print(f"📁 Sortie: {output}")

    return resultats


def traiter_dossier():
    input_dir = Path("dataset/input")
    output_dir = Path("dataset/processed")
    coefficients = charger_coefficients()

    cours_input_dir = input_dir / "cours"
    epreuves_input_dir = input_dir / "epreuves"
    cours_output_dir = output_dir / "cours"
    epreuves_output_dir = output_dir / "epreuves"

    resultats = []
    resultats.extend(
        traiter_type_documents(
            cours_input_dir,
            cours_output_dir,
            "cours",
            coefficients
        )
    )

    if epreuves_input_dir.exists():
        resultats.extend(
            traiter_type_documents(
                epreuves_input_dir,
                epreuves_output_dir,
                "epreuve",
                coefficients
            )
        )
    else:
        print(
            "ℹ️ dataset/input/epreuves absent, "
            "utilisation du layout historique dans dataset/input"
        )
        resultats.extend(
            traiter_type_documents(
                input_dir,
                epreuves_output_dir,
                "epreuve",
                coefficients,
                exclus={"cours", "epreuves"}
            )
        )

    print(f"\n✅ {len(resultats)} documents traites")
    return resultats


if __name__ == "__main__":
    traiter_dossier()
