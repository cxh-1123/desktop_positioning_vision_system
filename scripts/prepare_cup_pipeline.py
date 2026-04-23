import json, random, shutil
from pathlib import Path

root = Path(r"D:/Cxh/desktop_positioning_vision_system")
raw = root / "raw_images" / "cup"
dataset_root = root / "dataset" / "cup"
img_exts = {".jpg", ".jpeg", ".png"}

images = sorted([p for p in raw.iterdir() if p.is_file() and p.suffix.lower() in img_exts]) if raw.exists() else []
label_for_image = {img: img.with_suffix('.txt') for img in images}
matched = [img for img, lbl in label_for_image.items() if lbl.exists()]
missing = [img for img, lbl in label_for_image.items() if not lbl.exists()]

sample_names = [p.name for p in images[:10]]
sample_with_label = [{"image": p.name, "label_exists": p.with_suffix('.txt').exists()} for p in images[:10]]

result = {
    "project_root": str(root),
    "raw_dir": str(raw),
    "image_count": len(images),
    "label_count": len(matched),
    "missing_label_count": len(missing),
    "sample_images": sample_names,
    "sample_image_label_status": sample_with_label,
    "dataset_prepared": False,
    "cup_yaml": None,
    "train_images": 0,
    "val_images": 0,
}

# If any matched labels exist, prepare dataset from valid image-label pairs
if matched:
    img_train = dataset_root / "images" / "train"
    img_val = dataset_root / "images" / "val"
    lbl_train = dataset_root / "labels" / "train"
    lbl_val = dataset_root / "labels" / "val"
    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    pairs = [(img, img.with_suffix('.txt')) for img in matched]
    random.seed(42)
    random.shuffle(pairs)

    if len(pairs) == 1:
        train_pairs, val_pairs = pairs, []
    else:
        n_train = max(1, int(len(pairs) * 0.8))
        if n_train >= len(pairs):
            n_train = len(pairs) - 1
        train_pairs, val_pairs = pairs[:n_train], pairs[n_train:]

    # Clean only files that this script manages in dataset split dirs
    for d in [img_train, img_val, lbl_train, lbl_val]:
        for f in d.iterdir():
            if f.is_file():
                f.unlink()

    for img, lbl in train_pairs:
        shutil.copy2(img, img_train / img.name)
        shutil.copy2(lbl, lbl_train / lbl.name)
    for img, lbl in val_pairs:
        shutil.copy2(img, img_val / img.name)
        shutil.copy2(lbl, lbl_val / lbl.name)

    cup_yaml = dataset_root / "cup.yaml"
    cup_yaml.write_text(
        "path: D:/Cxh/desktop_positioning_vision_system/dataset/cup\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: cup\n",
        encoding="utf-8"
    )

    result.update({
        "dataset_prepared": True,
        "cup_yaml": str(cup_yaml),
        "train_images": len(train_pairs),
        "val_images": len(val_pairs),
    })

# Always write README guidance if labels are missing or absent
if len(matched) == 0 or len(missing) > 0:
    readme = root / "README_cup_training.txt"
    lines = [
        "Desktop Positioning Vision System - cup detection prep\n",
        f"Project root: {root}\n",
        f"Current image directory: {raw}\n",
        f"Current image count: {len(images)}\n",
        f"Current detected YOLO label count: {len(matched)}\n",
        f"Current missing label count: {len(missing)}\n",
        "\n",
        "Current label discovery status:\n",
        ("- No usable YOLO labels found yet.\n" if len(matched)==0 else "- Partial labels found; some images are still missing labels.\n"),
        "\n",
        "Next step: annotate images in YOLO format\n",
        "1) Use an annotation tool (LabelImg/Label Studio/Roboflow/etc.).\n",
        "2) Define single class id 0 for cup.\n",
        "3) For each image, create a same-name .txt file in the same folder as image now, or later move them for training split.\n",
        "\n",
        "YOLO label format (one object per line):\n",
        "class_id x_center y_center width height\n",
        "- All coordinates are normalized to [0,1] relative to image size.\n",
        "- For this single-class task, class_id must be 0.\n",
        "Example:\n",
        "0 0.512 0.476 0.221 0.338\n",
        "\n",
        "Where to place labels after annotation:\n",
        f"- Recommended raw co-location: {raw} (same basename as image).\n",
        "\n",
        "How to continue training after labels are complete:\n",
        "- Re-run data prep/split to refresh dataset/cup/images and dataset/cup/labels.\n",
        "- Then run:\n",
        "  yolo detect train data=D:/Cxh/desktop_positioning_vision_system/dataset/cup/cup.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=auto project=D:/Cxh/desktop_positioning_vision_system/outputs/train_runs name=cup_exp\n",
    ]
    readme.write_text("".join(lines), encoding="utf-8")
    result["readme"] = str(readme)
else:
    result["readme"] = None

report = root / "scripts" / "cup_prep_report.json"
report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
