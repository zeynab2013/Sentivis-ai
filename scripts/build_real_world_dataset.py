"""Build real-world validation dataset from COCO val2017 photographs."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "validation" / "real_world"
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_URL = "https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_val2017.json"
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017/{file_name}"

# COCO category names aligned with YOLO/COCO-80
SCENE_SPECS: dict[str, dict[str, object]] = {
    "sports": {
        "match_labels": {"sports ball", "tennis racket", "baseball bat", "skateboard", "surfboard", "frisbee"},
        "prefer_labels": {"person", "sports ball"},
    },
    "indoor": {
        "match_labels": {"couch", "bed", "tv", "chair", "dining table", "potted plant"},
        "prefer_labels": {"couch", "tv"},
    },
    "outdoor": {
        "match_labels": {"bench", "bird", "dog", "potted plant", "umbrella", "kite"},
        "prefer_labels": {"bench", "bird"},
    },
    "streets": {
        "match_labels": {"car", "bus", "traffic light", "stop sign", "fire hydrant", "parking meter"},
        "prefer_labels": {"car", "traffic light"},
    },
    "vehicles": {
        "match_labels": {"car", "bus", "truck", "motorcycle", "bicycle", "train", "airplane"},
        "prefer_labels": {"car", "bus"},
    },
    "people": {
        "match_labels": {"person"},
        "prefer_labels": {"person"},
        "min_persons": 2,
    },
    "animals": {
        "match_labels": {"dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear"},
        "prefer_labels": {"dog", "cat"},
    },
    "kitchens": {
        "match_labels": {"oven", "refrigerator", "sink", "microwave", "toaster", "bowl", "bottle"},
        "prefer_labels": {"oven", "sink", "refrigerator"},
    },
    "offices": {
        "match_labels": {"laptop", "keyboard", "mouse", "cell phone", "book", "tv"},
        "prefer_labels": {"laptop", "keyboard"},
    },
    "classrooms": {
        "match_labels": {"person", "chair", "book", "laptop", "backpack"},
        "prefer_labels": {"person", "chair", "book"},
    },
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {dest.name}")
    urllib.request.urlretrieve(url, dest)


def _load_coco(root: Path) -> tuple[dict, dict, dict]:
    ann_path = root / "instances_val2017.json"
    _download(ANNOTATIONS_URL, ann_path)
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    img_id_to_info = {img["id"]: img for img in data["images"]}
    img_to_labels: dict[int, set[str]] = {}
    img_to_anns: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        label = cat_id_to_name[ann["category_id"]]
        img_to_labels.setdefault(img_id, set()).add(label)
        img_to_anns.setdefault(img_id, []).append(
            {
                "label": label,
                "bbox": ann["bbox"],  # x, y, w, h
                "area": ann["area"],
                "category_id": ann["category_id"],
            }
        )
    return img_id_to_info, img_to_labels, img_to_anns


def _score_image(labels: set[str], spec: dict[str, object], person_count: int = 0) -> float:
    match_labels = set(spec["match_labels"])  # type: ignore[arg-type]
    prefer_labels = set(spec.get("prefer_labels", ()))  # type: ignore[arg-type]
    overlap = labels & match_labels
    if not overlap:
        return -1.0
    min_persons = int(spec.get("min_persons") or 0)
    if min_persons and person_count < min_persons:
        return -1.0
    score = float(len(overlap)) + 0.5 * len(labels & prefer_labels)
    return score


def _select_images(
    img_id_to_info: dict,
    img_to_labels: dict[int, set[str]],
    img_to_anns: dict[int, list[dict]],
    *,
    per_scene: int = 2,
) -> list[dict]:
    chosen: list[dict] = []
    used_ids: set[int] = set()

    for scene_type, spec in SCENE_SPECS.items():
        candidates: list[tuple[float, int]] = []
        min_persons = int(spec.get("min_persons") or 0)
        for img_id, labels in img_to_labels.items():
            if img_id in used_ids:
                continue
            if min_persons:
                person_count = sum(1 for ann in img_to_anns.get(img_id, []) if ann["label"] == "person")
                if person_count < min_persons:
                    continue
            person_count = sum(1 for ann in img_to_anns.get(img_id, []) if ann["label"] == "person")
            score = _score_image(labels, spec, person_count)
            if score >= 0:
                candidates.append((score, img_id))
        candidates.sort(reverse=True)
        picked = 0
        for _, img_id in candidates:
            if picked >= per_scene:
                break
            info = img_id_to_info[img_id]
            labels = sorted(img_to_labels[img_id])
            anns = img_to_anns.get(img_id, [])
            chosen.append(
                {
                    "scene_type": scene_type,
                    "image_id": img_id,
                    "file_name": info["file_name"],
                    "width": info["width"],
                    "height": info["height"],
                    "coco_url": IMAGE_BASE_URL.format(file_name=info["file_name"]),
                    "ground_truth_labels": labels,
                    "ground_truth_objects": anns,
                }
            )
            used_ids.add(img_id)
            picked += 1
        if picked < per_scene:
            raise RuntimeError(f"Could not find {per_scene} images for scene type {scene_type}")
    return chosen


def build_dataset(*, per_scene: int = 2) -> Path:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    img_id_to_info, img_to_labels, img_to_anns = _load_coco(DATASET_DIR)
    manifest_items = _select_images(img_id_to_info, img_to_labels, img_to_anns, per_scene=per_scene)

    for item in manifest_items:
        dest = IMAGES_DIR / item["file_name"]
        _download(item["coco_url"], dest)
        item["local_path"] = str(dest.relative_to(ROOT)).replace("\\", "/")

    manifest = {
        "source": "COCO val2017 (real photographs)",
        "annotation_source": ANNOTATIONS_URL,
        "image_count": len(manifest_items),
        "images": manifest_items,
    }
    manifest_path = DATASET_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dataset ready: {len(manifest_items)} images -> {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    build_dataset()
