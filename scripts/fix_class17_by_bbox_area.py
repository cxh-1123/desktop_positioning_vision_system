"""Re-map legacy class-17 boxes by bbox area: large -> cable(2), small -> small_item(3)."""

from __future__ import annotations

import subprocess
from pathlib import Path

# w*h; gap between small/large clusters in this dataset (~0.02 vs ~0.05+)
AREA_THRESHOLD = 0.025

FILES_WITH_17 = [
    "cup_04",
    "cup_09",
    "cup_11",
    "test_IMG_5045",
    "test_IMG_5046",
    "test_IMG_5050",
    "test_IMG_5056",
    "test_IMG_5057",
    "test_IMG_5058",
    "test_IMG_5059",
    "test_IMG_5060",
    "test_IMG_5061",
    "test_IMG_5062",
    "test_IMG_5063",
    "test_IMG_5064",
    "test_IMG_5065",
    "test_IMG_5068",
    "test_IMG_5074",
    "test_IMG_5075",
    "test_IMG_5076",
    "test_IMG_5079",
    "test_IMG_5083",
    "test_IMG_5085",
]


def parse_box(line: str) -> tuple[int, tuple[float, float, float, float]] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    coords = tuple(float(x) for x in parts[1:5])
    return cls, coords


def coords_key(coords: tuple[float, float, float, float]) -> tuple[float, ...]:
    return tuple(round(c, 6) for c in coords)


def legacy_17_boxes_from_git(stem: str, repo: Path) -> dict[tuple[float, ...], int]:
    rel = f"dataset/multi/labels/train/{stem}.txt"
    r = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    if r.returncode != 0:
        raise FileNotFoundError(rel)

    mapping: dict[tuple[float, ...], int] = {}
    for line in r.stdout.splitlines():
        parsed = parse_box(line)
        if not parsed or parsed[0] != 17:
            continue
        _, coords = parsed
        area = coords[2] * coords[3]
        new_cls = 2 if area >= AREA_THRESHOLD else 3
        mapping[coords_key(coords)] = new_cls
    return mapping


def apply_fix(label_path: Path, target: dict[tuple[float, ...], int]) -> tuple[int, int, int]:
    lines = label_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    to_cable = to_small = 0

    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        parsed = parse_box(line)
        if not parsed:
            out.append(line)
            continue
        cls, coords = parsed
        key = coords_key(coords)
        if key in target:
            cls = target[key]
            if cls == 2:
                to_cable += 1
            else:
                to_small += 1
        parts = [str(cls), *[f"{c:.6f}".rstrip("0").rstrip(".") if "." in f"{c:.6f}" else f"{c}" for c in coords]]
        # keep original float formatting style from file
        parts = [str(cls)] + [str(c) for c in coords]
        out.append(" ".join(parts))

    label_path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    return to_cable, to_small, len(target)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    train_dir = repo / "dataset" / "multi" / "labels" / "train"
    total_cable = total_small = 0

    for stem in FILES_WITH_17:
        target = legacy_17_boxes_from_git(stem, repo)
        path = train_dir / f"{stem}.txt"
        c, s, n = apply_fix(path, target)
        total_cable += c
        total_small += s
        print(f"{stem}: {c} -> cable(2), {s} -> small_item(3) (of {n})")

    cache = train_dir.parent / "train.cache"
    if cache.exists():
        cache.unlink()
        print("removed train.cache")

    print(f"total: cable={total_cable}, small_item={total_small}, threshold={AREA_THRESHOLD}")


if __name__ == "__main__":
    main()
