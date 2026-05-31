"""Remap legacy train label class IDs 15-23 to current 7-class IDs 0-6."""

from __future__ import annotations

from pathlib import Path

# Legacy export used global IDs 15-23; current classes (classes.txt):
# 0 bottle, 1 box, 2 cable, 3 small_item, 4 daily_item, 5 book, 6 tool
REMAP: dict[int, int] = {
    15: 0,  # bottle (legacy cup / drink)
    16: 1,  # box
    17: 3,  # fallback; run fix_class17_by_bbox_area.py for area-based 2/3 split
    18: 3,  # small_item
    19: 4,  # daily_item
    20: 5,  # book
    21: 6,  # tool
    22: 5,  # book (large instances)
    23: 3,  # small_item (tiny instances)
}


def remap_file(path: Path) -> tuple[bool, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    changed = 0
    needs_remap = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append(line)
            continue
        parts = stripped.split()
        old = int(parts[0])
        if old > 6:
            needs_remap = True
        if old in REMAP:
            parts[0] = str(REMAP[old])
            changed += 1
        elif old > 6:
            raise ValueError(f"{path}: unmapped class id {old}")
        out.append(" ".join(parts))

    if needs_remap:
        path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    return needs_remap, changed


def main() -> None:
    train_dir = Path("dataset/multi/labels/train")
    remapped_files = 0
    total_boxes = 0

    for path in sorted(train_dir.glob("*.txt")):
        if path.stem == "classes":
            continue
        did, n = remap_file(path)
        if did:
            remapped_files += 1
            total_boxes += n

    print(f"remapped_files={remapped_files}")
    print(f"remapped_boxes={total_boxes}")


if __name__ == "__main__":
    main()
