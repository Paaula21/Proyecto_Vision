#!/usr/bin/env python3
"""Genera las cifras y figuras del análisis del train set de la Parte 1.

Uso (desde la raíz del repositorio):

    python analysis/make_analysis.py
    python analysis/make_analysis.py --data fruits_part1/train --out analysis

Escribe:
    analysis/metrics.json
    analysis/figures/*.png

No implementa ningún detector: solo lee annotations.json, images.json e imágenes
y calcula estadísticas del dataset.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# RLE de COCO
# ---------------------------------------------------------------------------

def decode_counts(counts: str) -> np.ndarray:
    """Decodifica la lista de longitudes comprimida del RLE de COCO."""
    out: list[int] = []
    p = 0
    n = len(counts)
    while p < n:
        x = 0
        k = 0
        more = 1
        while more:
            c = ord(counts[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(out) > 2:
            x += out[-2]
        out.append(x)
    return np.asarray(out, dtype=np.int64)


def decode_rle(seg: dict) -> np.ndarray:
    """Máscara booleana (alto, ancho) a partir de un RLE de COCO."""
    h, w = seg["size"]
    counts = decode_counts(seg["counts"])
    if int(counts.sum()) != h * w:
        raise ValueError(f"RLE incoherente: suma {counts.sum()} != {h * w}")
    vals = np.zeros(len(counts), dtype=np.uint8)
    vals[1::2] = 1
    flat = np.repeat(vals, counts)
    return flat.reshape((w, h)).T.astype(bool)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def union_find_components(edges: dict[int, set[int]], ids: list[int]) -> list[list[int]]:
    parent = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for images in edges.values():
        images = list(images)
        for other in images[1:]:
            union(images[0], other)

    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    return sorted((sorted(v) for v in groups.values()), key=min)


def mask_component_stats(mask: np.ndarray, bbox: list[float]) -> tuple[int, int]:
    """(nº de piezas, tamaño de la pieza más pequeña) con conectividad 8."""
    x, y, w, h = (int(round(v)) for v in bbox)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(mask.shape[1], x + w), min(mask.shape[0], y + h)
    crop = mask[y0:y1, x0:x1]
    if crop.size == 0 or not crop.any():
        return 0, 0
    lab, n = ndimage.label(crop, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), int(sizes.min())


def summarize(values: list[float] | np.ndarray) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p10": float(np.percentile(a, 10)),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def species_color(cid: int) -> tuple[int, int, int]:
    return {1: (255, 140, 0), 2: (220, 30, 30), 3: (160, 200, 40), 4: (240, 220, 0)}.get(
        cid, (255, 255, 255)
    )


def outline(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_dilation(mask, iterations=1) & ~mask


def crop_with_outline(
    image: Image.Image,
    mask: np.ndarray,
    bbox: list[float],
    pad: int,
    scale: int,
    color: tuple[int, int, int],
) -> Image.Image:
    W, H = image.size
    x, y, w, h = (int(round(v)) for v in bbox)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    arr = np.asarray(image.crop((x0, y0, x1, y1))).copy()
    m = mask[y0:y1, x0:x1]
    arr[outline(m)] = color
    return Image.fromarray(arr).resize(((x1 - x0) * scale, (y1 - y0) * scale), Image.NEAREST)


def label_panel(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 22), (255, 255, 255))
    out.paste(img, (0, 22))
    ImageDraw.Draw(out).text((4, 5), text, fill=(0, 0, 0))
    return out


def fit_panel(panel: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    scale = min(cell_w / panel.width, cell_h / panel.height)
    size = (max(1, int(round(panel.width * scale))), max(1, int(round(panel.height * scale))))
    return panel.resize(size, Image.LANCZOS)


def montage(panels: list[Image.Image], cols: int, pad: int = 8,
            cell: tuple[int, int] | None = None) -> Image.Image:
    rows = (len(panels) + cols - 1) // cols
    if cell is None:
        cell_w = max(p.width for p in panels)
        cell_h = max(p.height for p in panels)
    else:
        cell_w, cell_h = cell
        panels = [fit_panel(p, cell_w, cell_h) for p in panels]
    out = Image.new("RGB", (cols * cell_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad),
                    (255, 255, 255))
    for k, p in enumerate(panels):
        r, c = divmod(k, cols)
        x = pad + c * (cell_w + pad) + (cell_w - p.width) // 2
        y = pad + r * (cell_h + pad) + (cell_h - p.height) // 2
        out.paste(p, (x, y))
    return out


# ---------------------------------------------------------------------------
# Cálculo principal
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("fruits_part1/train"),
                    help="carpeta con images/, annotations.json e images.json")
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    data: Path = args.data
    out: Path = args.out
    figdir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    ann_path = data / "annotations.json"
    dataset = json.loads(ann_path.read_text())
    images_info = json.loads((data / "images.json").read_text())
    anns = dataset["annotations"]
    images = dataset["images"]
    categories = {c["id"]: c["name"] for c in dataset["categories"]}
    image_by_id = {im["id"]: im for im in images}
    W = int(images[0]["width"])
    H = int(images[0]["height"])

    visible = [a for a in anns if not a["iscrowd"]]
    crowd = [a for a in anns if a["iscrowd"]]

    # --- especies -----------------------------------------------------------
    species: dict[str, dict] = {
        name: {"id": cid, "images": set(), "annotations": 0, "visible": 0,
               "ignored": 0, "fruit_ids": set()}
        for cid, name in categories.items()
    }
    for a in anns:
        s = species[categories[a["category_id"]]]
        s["annotations"] += 1
        s["fruit_ids"].add(a["fruit_id"])
        if a["iscrowd"]:
            s["ignored"] += 1
        else:
            s["visible"] += 1
        s["images"].add(a["image_id"])

    # --- grupos de árboles (grafo por fruit_id) -----------------------------
    fruit_images: dict[int, set[int]] = collections.defaultdict(set)
    fruit_categories: dict[int, set[int]] = collections.defaultdict(set)
    for a in anns:
        fruit_images[a["fruit_id"]].add(a["image_id"])
        fruit_categories[a["fruit_id"]].add(a["category_id"])
    components = union_find_components(fruit_images, sorted(image_by_id))
    trees = []
    for k, ids in enumerate(components):
        idset = set(ids)
        sub = [a for a in anns if a["image_id"] in idset]
        species_count = collections.Counter(categories[a["category_id"]] for a in sub)
        trees.append({
            "id": k,
            "image_ids": ids,
            "n_images": len(ids),
            "species": species_count.most_common(1)[0][0],
            "species_counts": dict(species_count),
            "fruit_ids": len({a["fruit_id"] for a in sub}),
            "annotations": len(sub),
            "visible": sum(1 for a in sub if not a["iscrowd"]),
            "ignored": sum(1 for a in sub if a["iscrowd"]),
        })

    # --- tamaños ------------------------------------------------------------
    areas = np.asarray([a["area"] for a in visible], dtype=float)
    bw = np.asarray([a["bbox"][2] for a in visible], dtype=float)
    bh = np.asarray([a["bbox"][3] for a in visible], dtype=float)
    small = int((areas < 32 * 32).sum())
    medium = int(((areas >= 32 * 32) & (areas < 96 * 96)).sum())
    large = int((areas >= 96 * 96).sum())

    # --- densidad -----------------------------------------------------------
    per_img_all = collections.Counter(a["image_id"] for a in anns)
    per_img_vis = collections.Counter(a["image_id"] for a in visible)

    # --- oclusión, piezas e iscrowd (una pasada por imagen) -----------------
    by_img_vis: dict[int, list[dict]] = collections.defaultdict(list)
    by_img_crowd: dict[int, list[dict]] = collections.defaultdict(list)
    for a in visible:
        by_img_vis[a["image_id"]].append(a)
    for a in crowd:
        by_img_crowd[a["image_id"]].append(a)

    pieces: list[int] = []
    multi_piece = 0
    min_piece_lt20 = 0
    fill_ratio: list[float] = []
    border_cut = 0
    overlaps: list[float] = []
    adjacent = 0

    for img_id in sorted(image_by_id):
        union = np.zeros((H, W), dtype=bool)
        for a in by_img_vis.get(img_id, []):
            mask = decode_rle(a["segmentation"])
            union |= mask
            n_pieces, smallest = mask_component_stats(mask, a["bbox"])
            pieces.append(n_pieces)
            if n_pieces >= 2:
                multi_piece += 1
            if smallest < 20:
                min_piece_lt20 += 1
            x, y, w, h = a["bbox"]
            fill_ratio.append(a["area"] / (w * h) if w * h else 0.0)
            if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
                border_cut += 1
        crowds = by_img_crowd.get(img_id, [])
        if crowds:
            union_d = ndimage.binary_dilation(union, iterations=1)
            for a in crowds:
                mask = decode_rle(a["segmentation"])
                total = int(mask.sum())
                inter = int((mask & union).sum())
                overlaps.append(inter / total if total else 0.0)
                if (mask & union_d).any():
                    adjacent += 1

    pieces_arr = np.asarray(pieces, dtype=float)
    overlaps_arr = np.asarray(overlaps, dtype=float)
    crowd_areas = np.asarray([a["area"] for a in crowd], dtype=float)

    # --- comprobaciones de integridad --------------------------------------
    checks = {
        "rle_area_matches_annotation": True,
        "fruit_id_has_one_category": all(len(v) == 1 for v in fruit_categories.values()),
        "all_images_covered_by_a_component": sum(len(t["image_ids"]) for t in trees) == len(images),
        "components_are_disjoint": len({i for t in trees for i in t["image_ids"]}) == len(images),
        "four_components_of_50": len(trees) == 4 and all(t["n_images"] == 50 for t in trees),
        "each_component_one_species": all(len(t["species_counts"]) == 1 for t in trees),
    }
    # Comprobación de area sobre una muestra (decodificar todo duplica el coste).
    rng = np.random.default_rng(0)
    sample = rng.choice(len(visible), size=min(300, len(visible)), replace=False)
    for idx in sample:
        a = visible[int(idx)]
        if int(decode_rle(a["segmentation"]).sum()) != int(a["area"]):
            checks["rle_area_matches_annotation"] = False
            break

    metrics = {
        "dataset": {
            "path": str(data),
            "images": len(images),
            "width": W,
            "height": H,
            "annotations": len(anns),
            "visible": len(visible),
            "ignored": len(crowd),
            "ignored_pct": round(100 * len(crowd) / len(anns), 2),
            "fruit_ids": len(fruit_images),
            "sha256_annotations": sha256(ann_path),
        },
        "species": {
            name: {
                "id": s["id"],
                "images": len(s["images"]),
                "annotations": s["annotations"],
                "visible": s["visible"],
                "ignored": s["ignored"],
                "fruit_ids": len(s["fruit_ids"]),
            }
            for name, s in species.items()
        },
        "trees": trees,
        "sizes": {
            "area_visible": summarize(areas),
            "bbox_width": summarize(bw),
            "bbox_height": summarize(bh),
            "small_lt_32x32": small,
            "medium_32x32_to_96x96": medium,
            "large_gt_96x96": large,
            "small_pct": round(100 * small / len(visible), 2),
            "medium_pct": round(100 * medium / len(visible), 2),
            "large_pct": round(100 * large / len(visible), 2),
        },
        "density": {
            "annotations_per_image": {
                "min": int(min(per_img_all.values())),
                "median": float(np.median(list(per_img_all.values()))),
                "mean": float(np.mean(list(per_img_all.values()))),
                "max": int(max(per_img_all.values())),
            },
            "visible_per_image": {
                "min": int(min(per_img_vis.values())),
                "median": float(np.median(list(per_img_vis.values()))),
                "mean": float(np.mean(list(per_img_vis.values()))),
                "max": int(max(per_img_vis.values())),
            },
        },
        "occlusion": {
            "multi_piece": multi_piece,
            "multi_piece_pct": round(100 * multi_piece / len(visible), 2),
            "pieces_per_mask": summarize(pieces_arr),
            "max_pieces": int(pieces_arr.max()),
            "with_piece_lt_20px": min_piece_lt20,
            "with_piece_lt_20px_pct": round(100 * min_piece_lt20 / len(visible), 2),
            "bbox_fill_median": float(np.median(fill_ratio)),
            "bbox_fill_mean": float(np.mean(fill_ratio)),
            "border_cut": border_cut,
            "border_cut_pct": round(100 * border_cut / len(visible), 2),
        },
        "iscrowd": {
            "count": len(crowd),
            "pct_of_annotations": round(100 * len(crowd) / len(anns), 2),
            "area": summarize(crowd_areas),
            "overlap_with_visible_mean": float(overlaps_arr.mean()) if overlaps_arr.size else None,
            "overlap_ge_50_pct": round(100 * float((overlaps_arr >= 0.5).mean()), 2) if overlaps_arr.size else None,
            "adjacent_after_1px_dilation_pct": round(100 * adjacent / len(overlaps), 2) if overlaps else None,
        },
        "checks": checks,
    }

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    if args.no_figures:
        print(f"métricas escritas en {out / 'metrics.json'}")
        return

    # --- figuras ------------------------------------------------------------
    plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 9})

    # 1. Especies
    names = [c for c in categories.values()]
    vis = [species[n]["visible"] for n in names]
    ign = [species[n]["ignored"] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(len(names))
    ax.bar(x - 0.2, vis, 0.4, label="visibles", color="#4c72b0")
    ax.bar(x + 0.2, ign, 0.4, label="iscrowd", color="#c44e52")
    ax.set_xticks(x, names)
    ax.set_ylabel("anotaciones")
    ax.set_title("Anotaciones por especie")
    ax.legend()
    for k, (v, i) in enumerate(zip(vis, ign)):
        ax.text(k - 0.2, v, str(v), ha="center", va="bottom", fontsize=7)
        ax.text(k + 0.2, i, str(i), ha="center", va="bottom", fontsize=7)
    fig.savefig(figdir / "fig_species.png")
    plt.close(fig)

    # 2. Áreas
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.logspace(0, np.log10(max(areas.max(), 10)), 60)
    ax.hist(areas, bins=bins, color="#4c72b0", edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.axvline(32 * 32, color="#c44e52", ls="--", lw=1, label="small/medium (32²)")
    ax.axvline(96 * 96, color="#8172b2", ls="--", lw=1, label="medium/large (96²)")
    ax.set_xlabel("área visible (px)")
    ax.set_ylabel("nº de frutas")
    ax.set_title("Distribución del área visible")
    ax.legend()
    fig.savefig(figdir / "fig_area_hist.png")
    plt.close(fig)

    # 3. Bbox
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    hb = ax.hexbin(np.maximum(bw, 1), np.maximum(bh, 1), gridsize=40, mincnt=1,
                   xscale="log", yscale="log", cmap="viridis")
    ax.set_xlabel("ancho del bbox (px)")
    ax.set_ylabel("alto del bbox (px)")
    ax.set_title("Tamaño del bbox (escala log)")
    fig.colorbar(hb, ax=ax, label="nº de frutas")
    fig.savefig(figdir / "fig_bbox_hexbin.png")
    plt.close(fig)

    # 4. Densidad
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.hist(list(per_img_vis.values()), bins=range(0, max(per_img_vis.values()) + 6, 5),
            color="#55a868", edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(list(per_img_vis.values())), color="#c44e52", ls="--",
               label=f"mediana = {np.median(list(per_img_vis.values())):.0f}")
    ax.set_xlabel("frutas visibles por imagen")
    ax.set_ylabel("nº de imágenes")
    ax.set_title("Densidad de frutas por imagen")
    ax.legend()
    fig.savefig(figdir / "fig_density.png")
    plt.close(fig)

    # 5. Piezas por máscara
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    max_p = int(pieces_arr.max())
    ax.hist(pieces_arr, bins=np.arange(0.5, max_p + 1.5, 1), color="#8172b2",
            edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("nº de piezas conexas (conectividad 8)")
    ax.set_ylabel("nº de frutas (log)")
    ax.set_title("Fragmentación de las máscaras visibles")
    fig.savefig(figdir / "fig_pieces.png")
    plt.close(fig)

    # 6. Relleno del bbox
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.hist(fill_ratio, bins=np.linspace(0, 1, 41), color="#c44e52",
            edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(fill_ratio), color="black", ls="--",
               label=f"mediana = {np.median(fill_ratio):.2f}")
    ax.set_xlabel("área de la máscara / área del bbox")
    ax.set_ylabel("nº de frutas")
    ax.set_title("Relleno del bbox (oclusión e irregularidad)")
    ax.legend()
    fig.savefig(figdir / "fig_bbox_fill.png")
    plt.close(fig)

    # 7. Árboles
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    labels = [f"árbol {t['id']}\n{t['species']}" for t in trees]
    vals = [t["visible"] for t in trees]
    ax.bar(labels, vals, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"][:len(trees)])
    for k, v in enumerate(vals):
        ax.text(k, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("frutas visibles")
    ax.set_title("Frutas visibles por árbol (50 imágenes cada uno)")
    fig.savefig(figdir / "fig_trees.png")
    plt.close(fig)

    # 8. Ejemplos por especie
    species_panels = []
    for name in names:
        tree = next(t for t in trees if t["species"] == name)
        # Elige la imagen más cercana del árbol (mayor área mediana de fruta).
        img_id = max(
            tree["image_ids"],
            key=lambda i: float(np.median([a["area"] for a in by_img_vis.get(i, [])]))
            if by_img_vis.get(i) else 0.0,
        )
        im = Image.open(data / "images" / image_by_id[img_id]["file_name"]).convert("RGB")
        arr = np.asarray(im).copy()
        for a in by_img_vis.get(img_id, []):
            mask = decode_rle(a["segmentation"])
            arr[outline(mask)] = (255, 0, 0)
        panel = Image.fromarray(arr).resize((420, 315), Image.LANCZOS)
        species_panels.append(label_panel(panel, f"{name} · imagen {img_id}"))
    montage(species_panels, cols=2).save(figdir / "fig_examples_species.png")

    # 9. Ejemplos difíciles
    hard_panels = []
    # (a) fruta con más piezas
    cand = [a for a in visible if a["area"] >= 400]
    multi_a = max(cand, key=lambda a: mask_component_stats(decode_rle(a["segmentation"]), a["bbox"])[0])
    m = decode_rle(multi_a["segmentation"])
    im = Image.open(data / "images" / image_by_id[multi_a["image_id"]]["file_name"]).convert("RGB")
    p = crop_with_outline(im, m, multi_a["bbox"], 30, 4, (255, 0, 0))
    hard_panels.append(label_panel(p, f"multipieza ({mask_component_stats(m, multi_a['bbox'])[0]} piezas)"))
    # (b) fruta cortada por el borde (la de mayor área para que se vea bien)
    border_a = max(
        (a for a in visible
         if a["bbox"][0] <= 1 or a["bbox"][1] <= 1
         or a["bbox"][0] + a["bbox"][2] >= W - 1
         or a["bbox"][1] + a["bbox"][3] >= H - 1),
        key=lambda a: a["area"],
    )
    m = decode_rle(border_a["segmentation"])
    im = Image.open(data / "images" / image_by_id[border_a["image_id"]]["file_name"]).convert("RGB")
    p = crop_with_outline(im, m, border_a["bbox"], 30, 4, (255, 0, 0))
    hard_panels.append(label_panel(p, "cortada por el borde"))
    # (c) anotación ignorada de mayor área
    crowd_a = max(crowd, key=lambda a: a["area"])
    m = decode_rle(crowd_a["segmentation"])
    im = Image.open(data / "images" / image_by_id[crowd_a["image_id"]]["file_name"]).convert("RGB")
    p = crop_with_outline(im, m, crowd_a["bbox"], 15, 8, (255, 220, 0))
    hard_panels.append(label_panel(p, f"iscrowd ({crowd_a['area']} px)"))
    # (d) fruta visible diminuta
    tiny_a = min((a for a in visible if a["area"] >= 20), key=lambda a: abs(a["area"] - 60))
    m = decode_rle(tiny_a["segmentation"])
    im = Image.open(data / "images" / image_by_id[tiny_a["image_id"]]["file_name"]).convert("RGB")
    p = crop_with_outline(im, m, tiny_a["bbox"], 15, 8, (255, 0, 0))
    hard_panels.append(label_panel(p, f"fruta visible de {tiny_a['area']} px"))
    montage(hard_panels, cols=2, cell=(380, 380)).save(figdir / "fig_examples_hard.png")

    print(f"métricas en {out / 'metrics.json'}")
    print(f"figuras en {figdir}")
    print(json.dumps({k: metrics[k] for k in ("dataset", "sizes", "occlusion", "iscrowd", "checks")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
