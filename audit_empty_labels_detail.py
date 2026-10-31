#!/usr/bin/env python3
"""
Inventory empty YOLO detection label files for manual review.

Read-only with respect to the source dataset:
- Does not edit, delete, move, or relabel source images/labels.
- Treats an empty label file as an annotation state, NOT proof that an image
  contains no target object.
- Preserves manual review fields from an existing inventory CSV on reruns.

Requirements:
    pip install pyyaml pillow

Example (Windows):
    python audit_empty_labels_detail.py --data "C:\\LampuStres\\Re Zero\\data.yaml"
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MANUAL_FIELDS = ("review_status", "suspected_classes", "annotation_action", "review_notes")
DEFAULT_STATUS = "UNREVIEWED"
VALID_STATUSES = {"UNREVIEWED", "VALID_EMPTY", "MISSING_ANNOTATION", "UNCERTAIN", "EXCLUDED"}
FIELDS = (
    "image_path", "label_path", "split", "image_width", "image_height",
    "label_bytes", "review_status", "suspected_classes", "annotation_action",
    "review_notes",
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return data


def resolve_dataset_root(data_yaml: Path, cfg: dict) -> Path:
    raw = cfg.get("path")
    if raw:
        root = Path(str(raw))
        if not root.is_absolute():
            root = data_yaml.parent / root
    else:
        root = data_yaml.parent
    return root.resolve()


def expand_split_entries(entry, root: Path) -> list[Path]:
    """Resolve a YOLO split entry (directory, image, or text list)."""
    if entry is None:
        return []
    entries = entry if isinstance(entry, list) else [entry]
    found: list[Path] = []
    for item in entries:
        p = Path(str(item))
        if not p.is_absolute():
            p = root / p
        p = p.resolve()
        if p.is_dir():
            found.extend(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTS)
        elif p.is_file() and p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                q = Path(line)
                if not q.is_absolute():
                    q = (p.parent / q).resolve()
                if q.is_file() and q.suffix.lower() in IMAGE_EXTS:
                    found.append(q)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            found.append(p)
        else:
            print(f"[WARN] Split entry not found: {p}")
    # Stable de-duplication
    return list(dict.fromkeys(found))


def image_to_label(image_path: Path) -> Path:
    parts = list(image_path.parts)
    # Standard Ultralytics layout: .../images/<split>/x.jpg -> .../labels/<split>/x.txt
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    # Fallback for nonstandard layout: sibling labels directory.
    return image_path.parent.parent / "labels" / image_path.parent.name / (image_path.stem + ".txt")


def is_empty_label(label_path: Path) -> tuple[bool, int]:
    if not label_path.exists():
        return False, -1
    raw = label_path.read_text(encoding="utf-8-sig", errors="replace")
    valid_rows = 0
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cols = s.split()
        # YOLO detection row: class x_center y_center width height
        if len(cols) >= 5:
            try:
                int(float(cols[0]))
                vals = [float(v) for v in cols[1:5]]
                if all(math.isfinite(v) for v in vals):
                    valid_rows += 1
            except (ValueError, OverflowError):
                pass
    return valid_rows == 0, label_path.stat().st_size


def read_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return {r.get("image_path", ""): r for r in csv.DictReader(f) if r.get("image_path")}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheets(rows: list[dict], output_dir: Path, per_sheet: int = 40) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    thumb_w, thumb_h, label_h = 240, 150, 38
    cols = 4
    for start in range(0, len(rows), per_sheet):
        batch = rows[start:start + per_sheet]
        tile_h = thumb_h + label_h
        sheet = Image.new("RGB", (cols * thumb_w, math.ceil(len(batch) / cols) * tile_h), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, row in enumerate(batch):
            x = (idx % cols) * thumb_w
            y = (idx // cols) * tile_h
            p = Path(row["image_path"])
            try:
                with Image.open(p) as src:
                    im = src.convert("RGB")
                    im.thumbnail((thumb_w - 8, thumb_h - 8))
                    px = x + (thumb_w - im.width) // 2
                    py = y + (thumb_h - im.height) // 2
                    sheet.paste(im, (px, py))
            except Exception as exc:
                draw.text((x + 6, y + 8), f"IMAGE ERROR: {exc}", fill="red", font=font)
            name = f'{start + idx + 1}. {p.name}'
            # Include split and path tail to disambiguate repeated filenames.
            caption = f'{row["split"]} | {name}'
            draw.text((x + 5, y + thumb_h + 3), caption[:42], fill="black", font=font)
        out = output_dir / f"empty_labels_contact_{start // per_sheet + 1:03d}.jpg"
        sheet.save(out, quality=90)
        print(f"[OK] Contact sheet: {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a manual-review inventory of empty YOLO labels.")
    ap.add_argument("--data", required=True, help="Path to dataset YAML (e.g. data.yaml).")
    ap.add_argument("--output", default="dataset_audit/empty_labels_detail",
                    help="Output directory; defaults to dataset_audit/empty_labels_detail.")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    help="YAML split keys to inspect. Default: train val.")
    ap.add_argument("--no-contact-sheets", action="store_true", help="Skip contact sheet generation.")
    args = ap.parse_args()

    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    cfg = load_yaml(data_yaml)
    root = resolve_dataset_root(data_yaml, cfg)
    output = Path(args.output).expanduser().resolve()
    existing = read_existing(output / "empty_labels_inventory.csv")

    rows: list[dict] = []
    missing_label_count = 0
    for split in args.splits:
        images = expand_split_entries(cfg.get(split), root)
        print(f"[INFO] {split}: {len(images)} images discovered")
        for image_path in images:
            label_path = image_to_label(image_path)
            empty, label_bytes = is_empty_label(label_path)
            if not label_path.exists():
                missing_label_count += 1
                continue
            if not empty:
                continue
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except Exception:
                width, height = "", ""

            key = str(image_path)
            old = existing.get(key, {})
            status = old.get("review_status") or DEFAULT_STATUS
            if status not in VALID_STATUSES:
                status = DEFAULT_STATUS
            row = {
                "image_path": key,
                "label_path": str(label_path),
                "split": split,
                "image_width": width,
                "image_height": height,
                "label_bytes": label_bytes,
                "review_status": status,
                "suspected_classes": old.get("suspected_classes", ""),
                "annotation_action": old.get("annotation_action", ""),
                "review_notes": old.get("review_notes", ""),
            }
            rows.append(row)

    rows.sort(key=lambda r: (r["split"], r["image_path"].lower()))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "empty_labels_inventory.csv", rows)

    for split in args.splits:
        write_csv(output / f"empty_{split}_inventory.csv", [r for r in rows if r["split"] == split])

    status_counts = Counter(r["review_status"] for r in rows)
    split_counts = Counter(r["split"] for r in rows)
    summary_path = output / "empty_labels_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["empty_label_images_total", len(rows)])
        writer.writerow(["images_with_missing_label_file", missing_label_count])
        for split, count in sorted(split_counts.items()):
            writer.writerow([f"empty_labels_{split}", count])
        for status, count in sorted(status_counts.items()):
            writer.writerow([f"status_{status}", count])

    if not args.no_contact_sheets and rows:
        make_contact_sheets(rows, output / "contact_sheets")

    print("\\n===== AUDIT INVENTORY SUMMARY =====")
    print(f"Dataset root       : {root}")
    print(f"Empty label images : {len(rows)}")
    print(f"Missing label files: {missing_label_count}")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")
    print(f"Output directory  : {output}")
    print("Source dataset was not modified.")
    print("NOTE: empty label != confirmed negative image; manual visual review is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
