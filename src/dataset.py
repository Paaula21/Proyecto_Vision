"""Lectura y validación del split y de las anotaciones COCO de frutas."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


CATEGORIES = {1: "orange", 2: "apple", 3: "pear", 4: "lemon"}

"""
Calcula la huella de las anotaciones. 
Comprobamos que el split  fue generado con el mismo annotations.json que estamos usando
"""
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

"Lee los ids de los archivos que se han pasado. "
"Comprueba que no esté vacío"
def read_ids(path):
    path = Path(path)
    ids = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"La lista está vacía o contiene IDs duplicados: {path}")
    return ids

"Carga las listas de entrenamiento y validación del fold elegido."
def load_split(data_dir, splits_dir, fold):
    "Rechaza listas obsoletas, imágenes compartidas o fugas de fruit_id."
    data_dir, splits_dir = Path(data_dir), Path(splits_dir)
    annotations_path = data_dir / "annotations.json"
    checks = json.loads((splits_dir / "checks.json").read_text(encoding="utf-8"))
    folds = json.loads((splits_dir / "folds.json").read_text(encoding="utf-8"))
    annotations_hash = sha256(annotations_path)
    "Si no pasa, salta un error"
    if not checks.get("passed"):
        raise ValueError("El split no ha superado las comprobaciones; revisa checks.json")
    if checks.get("sha256_annotations") != annotations_hash or folds.get("sha256_annotations") != annotations_hash:
        raise ValueError("El split no corresponde al annotations.json actual")

    matching = [item for item in folds["folds"] if item["fold"] == fold]
    "Caso en el que encuentre más de 1 fold"
    if len(matching) != 1:
        raise ValueError(f"No se ha encontrado un único fold {fold}")
    fold_info = matching[0]
    fold_dir = splits_dir / f"fold_{fold}"
    train_ids = read_ids(fold_dir / "train_ids.txt")
    val_ids = read_ids(fold_dir / "val_ids.txt")
    if sorted(train_ids) != sorted(fold_info["train_ids"]) or sorted(val_ids) != sorted(fold_info["val_ids"]):
        raise ValueError("Los IDs no coinciden con folds.json; revisa o regenera el split")

    train_set, val_set = set(train_ids), set(val_ids)
    "Comprobamos que los ids no se repiten en ambos campos"
    if train_set & val_set:
        raise ValueError("Entrenamiento y validación se solapan")
    with annotations_path.open(encoding="utf-8") as file:
        dataset = json.load(file)
    image_ids = {image["id"] for image in dataset["images"]}
    
    "Comprobamos que estén todas las imágenes cubiertas"
    if train_set | val_set != image_ids:
        raise ValueError("Las listas no cubren exactamente las imágenes del dataset")
    train_fruits = {a["fruit_id"] for a in dataset["annotations"] if a["image_id"] in train_set}
    val_fruits = {a["fruit_id"] for a in dataset["annotations"] if a["image_id"] in val_set}
    if train_fruits & val_fruits:
        raise ValueError("Hay frutas compartidas entre entrenamiento y validación")
    return train_ids, val_ids, fold_info, annotations_hash

"Es el lector de imágenes y anotaciones para PyTorch"
class FruitDataset(Dataset):
    """
    Las anotaciones iscrowd se incluyen como positivos en este entrenamiento;
    el evaluador oficial sigue ignorándolas.
    """
    
    "Carga annotations.json, guarda los ids solicitados y comprueba las categorías"
    def __init__(self, data_dir, image_ids):
        self.data_dir = Path(data_dir)
        self.image_ids = list(image_ids)
        self.coco = COCO(str(self.data_dir / "annotations.json"))
        categories = {c["id"]: c["name"] for c in self.coco.dataset["categories"]}
        if categories != CATEGORIES:
            raise ValueError(f"Categorías inesperadas: {categories}")
        unknown = set(self.image_ids) - set(self.coco.getImgIds())
        if unknown:
            raise ValueError(f"IDs de imagen desconocidos: {sorted(unknown)}")

    "Indica cuantas imágenes contienen el conjunto"
    def __len__(self):
        return len(self.image_ids)

    "Carga una imagen y prepara las frutas (cajas, categorías y máscara)"
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        info = self.coco.loadImgs([image_id])[0]
        with Image.open(self.data_dir / "images" / info["file_name"]) as image:
            image_array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        height, width = info["height"], info["width"]
        if image_array.shape[:2] != (height, width):
            raise ValueError(f"Dimensiones incorrectas en la imagen {image_id}")
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float().div(255.0)

        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))
        boxes, labels, masks = [], [], []
        for annotation in annotations:
            x, y, box_width, box_height = annotation["bbox"]
            if box_width <= 0 or box_height <= 0:
                raise ValueError(f"Caja inválida en anotación {annotation['id']}")
            boxes.append([x, y, x + box_width, y + box_height])
            labels.append(annotation["category_id"])
            masks.append(self.coco.annToMask(annotation))

        mask_array = np.stack(masks) if masks else np.zeros((0, height, width), dtype=np.uint8)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(mask_array, dtype=torch.uint8),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
        }
        return image_tensor, target

"""
Agrupa los datos para el entrenamiento
Necesario ya que cada imágen puede contener un número distinto de frutas
"""
def collate_fn(batch):
    return tuple(zip(*batch))
