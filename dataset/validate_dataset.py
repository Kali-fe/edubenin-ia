import json
from pathlib import Path
from collections import Counter


def valider():
    fichier = Path(
        "dataset/edubenin_dataset_v2.json"
    )
    if not fichier.exists():
        print("❌ Dataset non trouvé")
        return

    with open(fichier, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"📊 Total exemples: {len(dataset)}")

    ids = [ex["id"] for ex in dataset]
    doublons = len(ids) - len(set(ids))
    print(f"🔁 Doublons IDs: {doublons}")

    instructions = [
        ex.get("instruction", "")
        for ex in dataset
    ]
    instr_doublons = len(instructions) - \
        len(set(instructions))
    print(f"🔁 Instructions dupliquées: {instr_doublons}")

    matieres = Counter(
        ex.get("matiere") for ex in dataset
    )
    print("\n📚 Distribution matières:")
    for m, n in sorted(matieres.items()):
        print(f"  {m}: {n}")

    types_count = Counter(
        ex.get("type") for ex in dataset
    )
    print("\n📝 Distribution types:")
    for t, n in sorted(types_count.items()):
        print(f"  {t}: {n}")

    sans_criteres = sum(
        1 for ex in dataset
        if not ex.get("criteres_correction")
    )
    print(f"\n⚠️  Sans critères: {sans_criteres}")

    print("\n✅ Validation terminée")


if __name__ == "__main__":
    valider()
