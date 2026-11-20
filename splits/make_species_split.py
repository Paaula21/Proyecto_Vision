"""Genera 160/40 imágenes estratificadas por especie, con solapamiento medido."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

if __package__:
    from .protocols import SPECIES_HOLDOUT
else:
    from protocols import SPECIES_HOLDOUT


def make_split(dataset, seed):
    categories = {item["id"]: item["name"] for item in dataset["categories"]}
    if categories != {1: "orange", 2: "apple", 3: "pear", 4: "lemon"}:
        raise ValueError("Se esperaban las cuatro categorías oficiales")
    image_ids = [item["id"] for item in dataset["images"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Hay IDs de imagen duplicados")
    species_by_image = {image_id: set() for image_id in image_ids}
    for annotation in dataset["annotations"]:
        species_by_image[annotation["image_id"]].add(annotation["category_id"])
    groups = defaultdict(list)
    for image_id, species in species_by_image.items():
        if len(species) != 1 or not species <= categories.keys():
            raise ValueError(f"La imagen {image_id} no tiene una única especie oficial")
        groups[next(iter(species))].append(image_id)
    if set(groups) != set(categories) or any(len(ids) != 50 for ids in groups.values()):
        raise ValueError("Se esperaban 50 imágenes de cada especie")

    rng = random.Random(seed)
    train_ids, val_ids, counts = [], [], {}
    for category_id in sorted(groups):
        ids = sorted(groups[category_id])
        selected = set(rng.sample(ids, 10))
        val_ids.extend(selected)
        train_ids.extend(image_id for image_id in ids if image_id not in selected)
        counts[categories[category_id]] = {"train": 40, "validation": 10}
    train_ids, val_ids = sorted(train_ids), sorted(val_ids)
    train_set, val_set = set(train_ids), set(val_ids)
    train_fruits = {a["fruit_id"] for a in dataset["annotations"] if a["image_id"] in train_set}
    val_fruits = {a["fruit_id"] for a in dataset["annotations"] if a["image_id"] in val_set}
    shared = sorted(train_fruits & val_fruits)
    return {
        "fold": 0,
        "val_species": [categories[key] for key in sorted(categories)],
        "train_ids": train_ids,
        "val_ids": val_ids,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "species_counts": counts,
        "n_train_fruit_ids": len(train_fruits),
        "n_val_fruit_ids": len(val_fruits),
        "n_shared_fruit_ids": len(shared),
        "shared_fruit_ids": shared,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("fruits_part1/train"))
    parser.add_argument("--out", type=Path, default=Path(".artifacts/splits/species_holdout"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out debe ser un directorio nuevo para conservar los splits existentes")
    path = args.data / "annotations.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    fold = make_split(dataset, args.seed)
    metadata = {
        "protocol": SPECIES_HOLDOUT,
        "unit": "image",
        "seed": args.seed,
        "sha256_annotations": hashlib.sha256(path.read_bytes()).hexdigest(),
        "data": str(args.data),
        "warnings": [
            "Entrenamiento y validación comparten árboles y fruit_id; esta prueba "
            "es solo diagnóstica y sus métricas de validación son optimistas."
        ],
    }
    args.out.mkdir(parents=True)
    folder = args.out / "fold_0"
    folder.mkdir()
    for name in ("train_ids", "val_ids"):
        (folder / f"{name}.txt").write_text("".join(f"{i}\n" for i in fold[name]), encoding="utf-8")
    for name, value in (
        ("folds.json", {**metadata, "n_folds": 1, "folds": [fold]}),
        ("checks.json", {**metadata, "passed": not fold["shared_fruit_ids"],
                         "structural_passed": True, "diagnostic_only": bool(fold["shared_fruit_ids"]),
                         "no_fruit_leak": not fold["shared_fruit_ids"],
                         "errors": ([f"{fold['n_shared_fruit_ids']} fruit_id aparecen en train y val"]
                                    if fold["shared_fruit_ids"] else []),
                         "species_counts": fold["species_counts"],
                         "n_shared_fruit_ids": fold["n_shared_fruit_ids"]}),
    ):
        (args.out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "species_counts": fold["species_counts"],
                      "n_shared_fruit_ids": fold["n_shared_fruit_ids"], "warnings": metadata["warnings"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
