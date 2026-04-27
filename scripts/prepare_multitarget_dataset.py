import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect_images(raw_root: Path) -> list[Path]:
    images: list[Path] = []
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
    return sorted(images)


def safe_name(path: Path, raw_root: Path, used: set[str]) -> str:
    rel = path.relative_to(raw_root)
    stem = "_".join(rel.with_suffix("").parts)
    ext = path.suffix.lower()
    candidate = f"{stem}{ext}"
    idx = 1
    while candidate in used:
        candidate = f"{stem}_{idx}{ext}"
        idx += 1
    used.add(candidate)
    return candidate


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def write_yaml(path: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {path.parent.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO dataset split for multi-target desktop detection."
    )
    parser.add_argument(
        "--root",
        default="D:/Cxh/desktop_positioning_vision_system",
        help="Project root directory.",
    )
    parser.add_argument(
        "--raw-dir",
        default="raw_images",
        help="Relative path to raw image directory.",
    )
    parser.add_argument(
        "--out-dir",
        default="dataset/multi",
        help="Relative path to output YOLO dataset directory.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )
    parser.add_argument(
        "--classes",
        default="cup,bottle,box,cable,stationery,tool",
        help="Comma separated class names for yaml.",
    )
    parser.add_argument(
        "--create-empty-labels",
        action="store_true",
        help="Create empty label txt placeholders when source labels are missing.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    raw_root = (root / args.raw_dir).resolve()
    out_root = (root / args.out_dir).resolve()

    images = collect_images(raw_root)
    if not images:
        raise SystemExit(f"No images found under: {raw_root}")

    random.seed(args.seed)
    random.shuffle(images)

    val_count = int(round(len(images) * args.val_ratio))
    if val_count <= 0 and len(images) > 1:
        val_count = 1
    if val_count >= len(images):
        val_count = len(images) - 1

    val_images = images[:val_count]
    train_images = images[val_count:]

    img_train = out_root / "images" / "train"
    img_val = out_root / "images" / "val"
    lbl_train = out_root / "labels" / "train"
    lbl_val = out_root / "labels" / "val"
    for d in [img_train, img_val, lbl_train, lbl_val]:
        ensure_clean_dir(d)

    used_names: set[str] = set()
    copied_labels = 0
    missing_labels = 0
    missing_items: list[dict[str, str]] = []

    def copy_pair(image_paths: list[Path], img_dst: Path, lbl_dst: Path) -> None:
        nonlocal copied_labels, missing_labels
        for src_img in image_paths:
            dst_name = safe_name(src_img, raw_root, used_names)
            dst_img = img_dst / dst_name
            shutil.copy2(src_img, dst_img)

            src_lbl = src_img.with_suffix(".txt")
            dst_lbl = lbl_dst / f"{Path(dst_name).stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
                copied_labels += 1
            else:
                missing_labels += 1
                missing_items.append(
                    {
                        "image_source": str(src_img),
                        "image_in_dataset": str(dst_img),
                        "expected_label": str(dst_lbl),
                    }
                )
                if args.create_empty_labels:
                    dst_lbl.write_text("", encoding="utf-8")

    copy_pair(train_images, img_train, lbl_train)
    copy_pair(val_images, img_val, lbl_val)

    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    yaml_path = out_root / "multi.yaml"
    write_yaml(yaml_path, class_names)

    report = {
        "root": str(root),
        "raw_root": str(raw_root),
        "output_root": str(out_root),
        "total_images": len(images),
        "train_images": len(train_images),
        "val_images": len(val_images),
        "copied_labels": copied_labels,
        "missing_labels": missing_labels,
        "yaml": str(yaml_path),
        "classes": class_names,
        "create_empty_labels": args.create_empty_labels,
        "missing_label_examples": missing_items[:30],
    }
    report_path = root / "scripts" / "multi_dataset_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
