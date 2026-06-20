"""
EduBénin — Génération du dataset d'entraînement
Utilise GPT-5.5 via API BuildWithAfri pour générer des exemples
à partir des épreuves et cours collectés.
"""

import os
import json
import time
import re
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ── Chargement .env ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(PROJECT_DIR / ".env")

API_KEY  = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://build.lewisnote.com/v1")
MODEL    = os.getenv("OPENAI_MODEL", "gpt-5.5")
MAX_WORKERS = int(os.getenv("DATASET_MAX_WORKERS", "3"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=180)

# ── Chemins ──────────────────────────────────────────────────
EPREUVES_DIR  = BASE_DIR / "input" / "epreuves"
COURS_DIR     = BASE_DIR / "input" / "cours"
COEFF_FILE    = BASE_DIR / "coefficients.json"
OUTPUT_FILE   = BASE_DIR / "edubenin_dataset_v2.json"
CHECKPOINT_FILE = BASE_DIR / "edubenin_dataset_v2.checkpoint.json"
COURS_CACHE_FILE = BASE_DIR / "cours_cache.json"
SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".txt", ".md", ".doc"}

SYSTEM_PROMPT = """Tu es un expert en éducation béninoise, spécialisé dans les programmes officiels du Ministère de l'Enseignement Secondaire du Bénin (MESB). Tu connais parfaitement les programmes de 3ème, Terminale A, B, C et D."""

# ── Chargement coefficients ──────────────────────────────────
with open(COEFF_FILE, encoding="utf-8") as f:
    COEFFICIENTS = json.load(f)["classes"]

# ── Extraction texte selon format ────────────────────────────
def extraire_texte(path: Path, max_chars: int | None = None) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".docx":
            from docx import Document
            doc = Document(str(path))
            lignes = []
            total = 0
            for p in doc.paragraphs:
                if not p.text.strip():
                    continue
                lignes.append(p.text)
                total += len(p.text)
                if max_chars and total >= max_chars:
                    break
            return "\n".join(lignes)[:max_chars] if max_chars else "\n".join(lignes)
        elif suffix == ".pdf":
            return extraire_pdf(path, max_chars=max_chars)
        elif suffix in [".txt", ".md"]:
            texte = path.read_text(encoding="utf-8")
            return texte[:max_chars] if max_chars else texte
        elif suffix == ".doc":
            return extraire_doc_legacy(path)
        else:
            print(f"   ⚠️  Format non supporté : {path.name}")
            return ""
    except Exception as e:
        print(f"   ❌ Erreur lecture {path.name} : {e}")
        return ""


def extraire_doc_legacy(path: Path) -> str:
    """Extrait un .doc si un outil local compatible est installé."""
    commandes = [
        ["antiword", str(path)],
        ["soffice", "--headless", "--convert-to", "txt:Text", "--outdir", str(path.parent), str(path)],
        ["libreoffice", "--headless", "--convert-to", "txt:Text", "--outdir", str(path.parent), str(path)],
    ]
    for commande in commandes:
        try:
            resultat = subprocess.run(
                commande,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=60,
                check=False,
            )
            if commande[0] == "antiword" and resultat.stdout.strip():
                return resultat.stdout
            txt_path = path.with_suffix(".txt")
            if commande[0] != "antiword" and txt_path.exists():
                texte = txt_path.read_text(encoding="utf-8", errors="ignore")
                txt_path.unlink(missing_ok=True)
                if texte.strip():
                    return texte
        except Exception:
            continue
    print(f"   ⚠️  Impossible d'extraire le .doc legacy : {path.name}")
    return ""


def extraire_pdf(path: Path, max_chars: int | None = None) -> str:
    if max_chars:
        try:
            from pdfminer.high_level import extract_text
            texte = extract_text(str(path), maxpages=8) or ""
            if texte.strip():
                return texte[:max_chars]
        except Exception as e:
            print(f"   ⚠️  Extraction pdfminer limitée échouée pour {path.name}: {e}", flush=True)

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        texte = ""
        for page in reader.pages:
            texte += (page.extract_text() or "") + "\n"
            if max_chars and len(texte) >= max_chars:
                return texte[:max_chars]
        if texte.strip():
            return texte
    except Exception as e:
        print(f"   ⚠️  Extraction pypdf échouée pour {path.name}: {e}", flush=True)

    try:
        import pdfplumber
        texte = ""
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                texte += (page.extract_text() or "") + "\n"
                if max_chars and len(texte) >= max_chars:
                    return texte[:max_chars]
        return texte
    except Exception as e:
        print(f"   ❌ Extraction PDF échouée pour {path.name}: {e}", flush=True)
        return ""

# ── Nombre d'exemples selon coefficient ──────────────────────
def get_nb_exemples(classe: str, matiere: str):
    classe_norm = classe.replace("é", "e").replace("è", "e")
    coeff = None
    for cl, matieres in COEFFICIENTS.items():
        if cl.lower() == classe_norm.lower():
            for mat, c in matieres.items():
                if mat.lower() in matiere.lower() or matiere.lower() in mat.lower():
                    coeff = c
                    break
    if coeff is None:
        coeff = 2  # valeur par défaut
    if coeff >= 5:
        return 14, 6, 6, 2
    elif coeff >= 3:
        return 10, 4, 4, 2
    else:
        return 6, 2, 2, 2

# ── Chargement cours par classe/matière ──────────────────────
def charger_cours() -> dict:
    if COURS_CACHE_FILE.exists():
        try:
            with open(COURS_CACHE_FILE, encoding="utf-8") as f:
                cache = json.load(f)
            print(f"   Cache cours chargé : {COURS_CACHE_FILE.name}", flush=True)
            return {
                tuple(cle.split("|", 1)): textes
                for cle, textes in cache.items()
            }
        except Exception as e:
            print(f"   ⚠️  Cache cours ignoré : {e}", flush=True)

    index = {}
    if not COURS_DIR.exists():
        return index
    for fichier in COURS_DIR.rglob("*"):
        if fichier.is_file() and fichier.suffix.lower() in SUPPORTED_EXTENSIONS:
            print(f"   Cours: extraction {fichier.relative_to(COURS_DIR)}", flush=True)
            texte = extraire_texte(fichier, max_chars=12000)
            if texte:
                parties = fichier.relative_to(COURS_DIR).parts
                classe = parties[0] if len(parties) >= 3 else ""
                matiere = parties[1] if len(parties) >= 3 else fichier.parent.name
                cle = (normaliser(classe), normaliser(matiere))
                index.setdefault(cle, [])
                index[cle].append(texte)
    try:
        with open(COURS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"|".join(cle): textes for cle, textes in index.items()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"   ⚠️  Cache cours non écrit : {e}", flush=True)
    return index

# ── Trouver le cours correspondant ───────────────────────────
def trouver_cours(index_cours: dict, classe: str, matiere: str) -> str:
    cle = (normaliser(classe), normaliser(matiere))
    textes = index_cours.get(cle, [])
    if textes:
        return "\n\n".join(textes)[:6000]
    return ""


def normaliser(valeur: str) -> str:
    return (
        valeur.lower()
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def prompt_regles(classe: str, matiere: str, nb: int, nb_expl: int, nb_exo: int, nb_dial: int) -> str:
    return f"""Génère exactement {nb} exemples d'entraînement répartis ainsi :
- {nb_expl} exemples de type "explication"
- {nb_exo} exemples de type "exercice"
- {nb_dial} exemples de type "dialogue"

FORMAT JSON pour chaque exemple :
{{
  "type": "explication" | "exercice" | "dialogue",
  "classe": "{classe}",
  "matiere": "{matiere}",
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

RÈGLES ABSOLUES pour EduBot :
- Toujours en français
- Toujours des exemples béninois (Dantokpa, Cotonou, Porto-Novo, FCFA, noms locaux)
- Ne jamais donner la réponse directement
- Guider par questions progressives
- Encourager l'élève à chaque étape
- Adapter le vocabulaire au niveau de la classe
- S'appuyer sur le cours fourni quand disponible
- Les dialogues doivent contenir au moins 4 messages alternés user/assistant
- Aucun texte avant ou après le JSON. Aucun bloc markdown."""

# ── Appel GPT-5.5 avec corrigé disponible ───────────────────
def generer_avec_corrige(epreuve: str, corrige: str, cours: str,
                          classe: str, matiere: str,
                          nb: int, nb_expl: int, nb_exo: int, nb_dial: int) -> list:
    cours_section = cours if cours else "Aucun cours disponible."

    prompt = f"""À partir de l'épreuve et du corrigé fournis, et du contenu de cours si disponible, génère des exemples d'entraînement.

{prompt_regles(classe, matiere, nb, nb_expl, nb_exo, nb_dial)}

Réponds UNIQUEMENT avec un tableau JSON brut [ ... ]

ÉPREUVE + CORRIGÉ :
ÉPREUVE :
{epreuve[:7000]}

CORRIGÉ :
{corrige[:5000]}

CONTENU DE COURS (si disponible) :
{cours_section}"""

    return appeler_gpt(prompt)

# ── Appel GPT-5.5 sans corrigé ───────────────────────────────
def generer_sans_corrige(epreuve: str, cours: str,
                          classe: str, matiere: str,
                          nb: int, nb_expl: int, nb_exo: int, nb_dial: int) -> tuple:
    cours_section = f"\n\nCONTENU DE COURS (si disponible) :\n{cours}" if cours else ""

    prompt = f"""L'épreuve suivante n'a pas de corrigé disponible.
Génère d'abord un corrigé complet et détaillé selon le programme officiel béninois, puis génère les exemples d'entraînement.

{prompt_regles(classe, matiere, nb, nb_expl, nb_exo, nb_dial)}

ÉPREUVE :
{epreuve[:7000]}
{cours_section}

Réponds avec un JSON structuré ainsi :
{{
  "corrige_genere": "corrigé complet ici",
  "exemples": [ ...tableau d'exemples... ]
}}"""

    for tentative in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8000
            )
            data = charger_json_reponse(response.choices[0].message.content)
            return valider_exemples(data.get("exemples", []), classe, matiere), True
        except Exception as e:
            print(f"   ⚠️  Tentative {tentative+1}/3 échouée : {e}")
            time.sleep(2)
    raise RuntimeError("Échec API après 3 tentatives pour génération avec corrigé auto")

# ── Appel GPT générique ──────────────────────────────────────
def appeler_gpt(prompt: str) -> list:
    for tentative in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8000
            )
            data = charger_json_reponse(response.choices[0].message.content)
            if not isinstance(data, list):
                raise ValueError("La réponse attendue doit être un tableau JSON")
            return data
        except Exception as e:
            print(f"   ⚠️  Tentative {tentative+1}/3 échouée : {e}")
            time.sleep(2)
    raise RuntimeError("Échec API après 3 tentatives")


def charger_json_reponse(texte: str):
    texte = (texte or "").strip()
    texte = re.sub(r"^```(?:json)?|```$", "", texte, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(texte)
    except json.JSONDecodeError:
        debut_obj = texte.find("{")
        debut_tab = texte.find("[")
        debuts = [i for i in [debut_obj, debut_tab] if i >= 0]
        if not debuts:
            raise
        debut = min(debuts)
        fin = max(texte.rfind("}"), texte.rfind("]"))
        if fin <= debut:
            raise
        extrait = texte[debut:fin + 1]
        try:
            return json.loads(extrait)
        except json.JSONDecodeError:
            # Dernier recours pour certaines réponses tronquées : récupérer les
            # objets exemples individuellement plutôt que bloquer tout le run.
            objets = re.findall(r"\{\s*\"type\"\s*:\s*\"(?:explication|exercice|dialogue)\".*?\}\s*(?=,\s*\{|\s*\])", extrait, flags=re.DOTALL)
            exemples = []
            for objet in objets:
                try:
                    exemples.append(json.loads(objet))
                except json.JSONDecodeError:
                    continue
            if exemples:
                return {"exemples": exemples} if extrait.lstrip().startswith("{") else exemples
            raise


def valider_exemples(exemples: list, classe: str, matiere: str) -> list:
    valides = []
    for exemple in exemples or []:
        if not isinstance(exemple, dict):
            continue
        if exemple.get("type") not in {"explication", "exercice", "dialogue"}:
            continue
        messages = exemple.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        exemple["classe"] = exemple.get("classe") or classe
        exemple["matiere"] = exemple.get("matiere") or matiere
        valides.append(exemple)
    return valides


def sauvegarder_checkpoint(tous_exemples: list, stats_run: dict, processed_sources: set[str], skipped_sources: set[str] | None = None) -> None:
    payload = {
        "exemples": tous_exemples,
        "stats": stats_run,
        "processed_sources": sorted(processed_sources),
        "skipped_sources": sorted(skipped_sources or set()),
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def charger_checkpoint() -> tuple[list, dict, set[str], set[str]]:
    if not CHECKPOINT_FILE.exists():
        return [], {}, set(), set()
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        payload = json.load(f)
    exemples = payload.get("exemples", [])
    stats_run = payload.get("stats", {})
    processed_sources = set(payload.get("processed_sources", []))
    skipped_sources = set(payload.get("skipped_sources", []))
    if processed_sources and not exemples:
        print("⚠️  Checkpoint vide détecté, reprise depuis le début", flush=True)
        return [], {}, set(), set()
    print(
        f"↩️  Reprise checkpoint: {len(processed_sources)} épreuves traitées, "
        f"{len(skipped_sources)} ignorées, "
        f"{len(exemples)} exemples déjà générés",
        flush=True,
    )
    return exemples, stats_run, processed_sources, skipped_sources

# ── Détection classe/matière depuis le chemin ────────────────
def detecter_classe_matiere(chemin: Path) -> tuple:
    parties = chemin.parts
    # Structure attendue : .../epreuves/Classe/Matiere/fichier
    try:
        idx = next(i for i, p in enumerate(parties) if p == "epreuves")
        classe  = parties[idx + 1] if idx + 1 < len(parties) else "Inconnue"
        matiere = parties[idx + 2] if idx + 2 < len(parties) else "Inconnue"
        return classe, matiere
    except StopIteration:
        return "Inconnue", "Inconnue"


def traiter_epreuve(epreuve_path: Path, index_cours: dict) -> dict:
    source_id = str(epreuve_path.relative_to(EPREUVES_DIR)).replace("\\", "/")
    classe, matiere = detecter_classe_matiere(epreuve_path)
    num = re.search(r"\d+", epreuve_path.stem)
    numero = num.group() if num else "?"

    corrige_path = epreuve_path.parent / f"Corrigé Epreuve{numero}{epreuve_path.suffix}"
    if not corrige_path.exists():
        for f in epreuve_path.parent.iterdir():
            if "corrig" in f.name.lower() and numero in f.name:
                corrige_path = f
                break
        else:
            corrige_path = None

    texte_epreuve = extraire_texte(epreuve_path)
    if not texte_epreuve:
        return {
            "source_id": source_id,
            "skipped": True,
            "message": f"   ⚠️  Épreuve vide, ignorée : {epreuve_path.name}",
        }

    cours = trouver_cours(index_cours, classe, matiere)
    nb, nb_expl, nb_exo, nb_dial = get_nb_exemples(classe, matiere)
    print(f"⏳ {classe} {matiere} Epreuve{numero} → génération de {nb} exemples...", flush=True)

    if corrige_path and corrige_path.exists():
        texte_corrige = extraire_texte(corrige_path)
        exemples = generer_avec_corrige(
            texte_epreuve, texte_corrige, cours,
            classe, matiere, nb, nb_expl, nb_exo, nb_dial
        )
        exemples = valider_exemples(exemples, classe, matiere)
        type_corrige = "original"
    else:
        exemples, _ = generer_sans_corrige(
            texte_epreuve, cours,
            classe, matiere, nb, nb_expl, nb_exo, nb_dial
        )
        type_corrige = "généré"

    if not exemples:
        raise RuntimeError(f"Aucun exemple généré pour {source_id}")

    return {
        "source_id": source_id,
        "skipped": False,
        "classe": classe,
        "matiere": matiere,
        "numero": numero,
        "exemples": exemples,
        "type_corrige": type_corrige,
    }

# ── Programme principal ──────────────────────────────────────
def main():
    print("=" * 60)
    print("EduBénin — Génération dataset v2")
    print("=" * 60)

    # Chargement index cours
    print("\n📚 Chargement des cours...")
    index_cours = charger_cours()
    print(f"   {len(index_cours)} fichiers de cours indexés")

    # Collecte de toutes les épreuves
    epreuves_files = [
        f for f in EPREUVES_DIR.rglob("*")
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_EXTENSIONS
        and "corrig" not in f.name.lower()
    ]

    print(f"\n📝 {len(epreuves_files)} épreuves trouvées\n")

    tous_exemples, checkpoint_stats, processed_sources, skipped_sources = charger_checkpoint()
    nb_corrige_genere = int(checkpoint_stats.get("corriges_generes", 0) or 0)
    nb_epreuves_traitees = int(checkpoint_stats.get("epreuves_traitees", 0) or 0)
    stats = checkpoint_stats.get("par_classe_matiere", {}) or {}
    stats_run = {
        "epreuves_traitees": nb_epreuves_traitees,
        "corriges_generes": nb_corrige_genere,
        "par_classe_matiere": stats,
    }

    remaining_files = []
    for epreuve_path in sorted(epreuves_files):
        source_id = str(epreuve_path.relative_to(EPREUVES_DIR)).replace("\\", "/")
        if source_id in processed_sources or source_id in skipped_sources:
            print(f"⏭️  {source_id} déjà traité, passage à la suivante", flush=True)
            continue
        remaining_files.append(epreuve_path)

    print(f"🚀 Génération parallèle: {len(remaining_files)} épreuves restantes, {MAX_WORKERS} workers", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(traiter_epreuve, path, index_cours): path for path in remaining_files}
        for future in as_completed(futures):
            future_path = futures[future]
            source_id_from_path = str(future_path.relative_to(EPREUVES_DIR)).replace("\\", "/")
            try:
                result = future.result()
            except Exception as e:
                skipped_sources.add(source_id_from_path)
                sauvegarder_checkpoint(tous_exemples, stats_run, processed_sources, skipped_sources)
                print(f"   ⚠️  Épreuve ignorée après échec : {source_id_from_path} ({e})", flush=True)
                continue

            source_id = result["source_id"]

            if result.get("skipped"):
                skipped_sources.add(source_id)
                sauvegarder_checkpoint(tous_exemples, stats_run, processed_sources, skipped_sources)
                print(result["message"], flush=True)
                continue

            classe = result["classe"]
            matiere = result["matiere"]
            numero = result["numero"]
            exemples = result["exemples"]
            type_corrige = result["type_corrige"]

            if type_corrige == "généré":
                nb_corrige_genere += 1

            nb_epreuves_traitees += 1
            tous_exemples.extend(exemples)

            cle_stat = f"{classe}/{matiere}"
            stats[cle_stat] = stats.get(cle_stat, 0) + len(exemples)
            stats_run["epreuves_traitees"] = nb_epreuves_traitees
            stats_run["corriges_generes"] = nb_corrige_genere
            processed_sources.add(source_id)
            sauvegarder_checkpoint(tous_exemples, stats_run, processed_sources, skipped_sources)

            print(f"✅ {classe} {matiere} Epreuve{numero} → {len(exemples)} exemples générés (corrigé: {type_corrige})", flush=True)

    # Sauvegarde
    tmp_output = OUTPUT_FILE.with_suffix(".tmp.json")
    with open(tmp_output, "w", encoding="utf-8") as f:
        json.dump(tous_exemples, f, ensure_ascii=False, indent=2)
    tmp_output.replace(OUTPUT_FILE)
    CHECKPOINT_FILE.unlink(missing_ok=True)

    # Résumé final
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"📊 Total épreuves traitées  : {nb_epreuves_traitees}")
    print(f"📊 Total exemples générés   : {len(tous_exemples)}")
    print(f"📊 Corrigés auto-générés    : {nb_corrige_genere}")
    print(f"💾 Dataset sauvegardé       : {OUTPUT_FILE}")
    print("\nRépartition par classe/matière :")
    for cle, nb in sorted(stats.items()):
        print(f"   {cle} → {nb} exemples")

if __name__ == "__main__":
    main()
