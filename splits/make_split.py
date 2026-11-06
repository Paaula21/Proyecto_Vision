#!/usr/bin/env python3
"""Construye y verifica el split leave-one-tree-out (LOTO) de la Parte 1.

Uso (desde la raíz del repositorio):

    python splits/make_split.py
    python splits/make_split.py --data fruits_part1/fruits_part1/train --out splits

El script:
1. Construye el grafo imagen-fruta-imagen usando `fruit_id`.
2. Obtiene los componentes conexos (árboles).
3. Comprueba que son 4 grupos de 50 imágenes, cada uno de una sola especie.
4. Genera 4 folds: validación = un árbol, entrenamiento = los otros tres.
5. Escribe folds.json, fold_k/train_ids.txt, fold_k/val_ids.txt y checks.json.
6. Falla con código distinto de 0 si alguna comprobación no se cumple.

No entrena nada ni modifica el dataset.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def components_from_fruit_graph(
    fruit_images: dict[int, set[int]], image_ids: list[int], min_views: int = 1
) -> list[list[int]]:
    """Componentes conexas del grafo de imágenes que comparten alguna fruta."""
    parent = {i: i for i in image_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for images in fruit_images.values():
        if len(images) < min_views:
            continue
        images = list(images)
        for other in images[1:]:
            union(images[0], other)

    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i in image_ids:
        groups[find(i)].append(i)
    return sorted((sorted(v) for v in groups.values()), key=min)


def write_ids(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{i}\n" for i in sorted(ids)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("fruits_part1/fruits_part1/train"),
                    help="carpeta con annotations.json e images.json")
    ap.add_argument("--out", type=Path, default=Path("splits"))
    args = ap.parse_args()

    data: Path = args.data
    out: Path = args.out
    ann_path = data / "annotations.json"
    dataset = json.loads(ann_path.read_text())
    images = dataset["images"]
    anns = dataset["annotations"]
    categories = {c["id"]: c["name"] for c in dataset["categories"]}
    image_ids = sorted(im["id"] for im in images)

    fruit_images: dict[int, set[int]] = collections.defaultdict(set)
    fruit_categories: dict[int, set[int]] = collections.defaultdict(set)
    for a in anns:
        fruit_images[a["fruit_id"]].add(a["image_id"])
        fruit_categories[a["fruit_id"]].add(a["category_id"])

    components = components_from_fruit_graph(fruit_images, image_ids)

    errors: list[str] = []
    warnings: list[str] = []

    if len(components) != 4:
        errors.append(
            f"se esperaban 4 componentes (árboles) y hay {len(components)}. "
            "No asignes componentes sueltos a folds: un mismo árbol podría quedar "
            "a ambos lados. Revisa la fusión y documenta la evidencia."
        )
    for k, comp in enumerate(components):
        if len(comp) != 50:
            errors.append(f"el componente {k} tiene {len(comp)} imágenes y se esperaban 50")
    for k, comp in enumerate(components):
        species = {categories[a["category_id"]] for a in anns if a["image_id"] in set(comp)}
        if len(species) != 1:
            errors.append(f"el componente {k} mezcla especies: {sorted(species)}")

    if any(len(v) != 1 for v in fruit_categories.values()):
        errors.append("hay fruit_id con más de una categoría")

    tree_rows = []
    for k, comp in enumerate(components):
        comp_set = set(comp)
        sub = [a for a in anns if a["image_id"] in comp_set]
        species = sorted({categories[a["category_id"]] for a in sub})
        tree_rows.append({
            "tree": k,
            "species": species[0] if len(species) == 1 else species,
            "n_images": len(comp),
            "image_ids": comp,
            "fruit_ids": len({a["fruit_id"] for a in sub}),
            "visible": sum(1 for a in sub if not a["iscrowd"]),
            "ignored": sum(1 for a in sub if a["iscrowd"]),
        })

    # Comprobación de robustez: reconstruir el grafo usando solo frutas vistas en
    # al menos N imágenes. Si el grupo se rompe, la separación por árbol sería
    # frágil y habría que documentarlo.
    robustness = {}
    for min_views in (1, 2, 3, 5, 8, 10, 15):
        comps = components_from_fruit_graph(fruit_images, image_ids, min_views=min_views)
        robustness[str(min_views)] = [len(c) for c in comps]

    folds = []
    if not errors:
        for k, val_comp in enumerate(components):
            val_set = set(val_comp)
            train_ids = [i for i in image_ids if i not in val_set]
            val_ids = sorted(val_comp)
            train_fruits = {a["fruit_id"] for a in anns if a["image_id"] in set(train_ids)}
            val_fruits = {a["fruit_id"] for a in anns if a["image_id"] in val_set}
            shared = train_fruits & val_fruits
            fold_errors = []
            if set(train_ids) & set(val_ids):
                fold_errors.append("train y val se solapan")
            if set(train_ids) | set(val_ids) != set(image_ids):
                fold_errors.append("train ∪ val no cubre todas las imágenes")
            if len(train_ids) != 150 or len(val_ids) != 50:
                fold_errors.append(f"tamaños inesperados: train={len(train_ids)}, val={len(val_ids)}")
            if shared:
                fold_errors.append(f"{len(shared)} fruit_id aparecen en train y val")
            species = tree_rows[k]["species"]
            folds.append({
                "fold": k,
                "val_tree": k,
                "val_species": species,
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "train_ids": sorted(train_ids),
                "val_ids": val_ids,
                "n_train_fruit_ids": len(train_fruits),
                "n_val_fruit_ids": len(val_fruits),
                "n_shared_fruit_ids": len(shared),
                "errors": fold_errors,
            })
            errors.extend(f"fold {k}: {e}" for e in fold_errors)
            write_ids(out / f"fold_{k}" / "train_ids.txt", train_ids)
            write_ids(out / f"fold_{k}" / "val_ids.txt", val_ids)

    passed = not errors

    if not passed:
        warnings.append("el split NO se ha publicado como válido; revisa errors")

    folds_doc = {
        "protocol": "leave-one-tree-out (LOTO)",
        "unit": "tree",
        "data": str(data),
        "sha256_annotations": sha256(ann_path),
        "n_folds": len(folds),
        "folds": folds,
    }
    checks = {
        "passed": passed,
        "data": str(data),
        "sha256_annotations": sha256(ann_path),
        "n_images": len(image_ids),
        "n_annotations": len(anns),
        "n_fruit_ids": len(fruit_images),
        "trees": tree_rows,
        "robustness_min_views": robustness,
        "folds": [
            {
                "fold": f["fold"],
                "val_tree": f["val_tree"],
                "val_species": f["val_species"],
                "n_train": f["n_train"],
                "n_val": f["n_val"],
                "n_train_fruit_ids": f["n_train_fruit_ids"],
                "n_val_fruit_ids": f["n_val_fruit_ids"],
                "n_shared_fruit_ids": f["n_shared_fruit_ids"],
                "train_val_disjoint": not bool(set(f["train_ids"]) & set(f["val_ids"])),
                "covers_all_images": set(f["train_ids"]) | set(f["val_ids"]) == set(image_ids),
                "no_fruit_leak": f["n_shared_fruit_ids"] == 0,
                "passed": not f["errors"],
                "errors": f["errors"],
            }
            for f in folds
        ],
        "errors": errors,
        "warnings": warnings,
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "folds.json").write_text(json.dumps(folds_doc, indent=2, ensure_ascii=False) + "\n")
    (out / "checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({"passed": passed, "errors": errors, "warnings": warnings,
                      "trees": [{"tree": t["tree"], "species": t["species"],
                                 "n_images": t["n_images"]} for t in tree_rows],
                      "folds": [{"fold": f["fold"], "val_species": f["val_species"],
                                 "n_train": f["n_train"], "n_val": f["n_val"]} for f in folds],
                      "robustness_min_views": robustness},
                     indent=2, ensure_ascii=False))

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
