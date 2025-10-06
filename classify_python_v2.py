#!/usr/bin/env python3
"""
Classify unlabeled PDFs in a folder (e.g., CCH) using prototypes
learned from the existing labeled training document (content.pdf + JSON).

Outputs
- Per-PDF predictions CSVs with columns: page, predicted_label
- Per-PDF ground-truth-like JSONs using predictions (document, config, total_pages, pages[])

Environment variables / CLI args:
- --input_dir / INPUT_DIR: directory of PDFs to classify (default: CCH)
- --train_pdf / TRAIN_PDF_PATH: training PDF path (default: content.pdf)
- --train_gt / TRAIN_GT_PATH: training GT JSON path (default: 23I_21PALCOACCT_V1_enhanced.json)
- --out_dir / OUTPUT_DIR: output directory for predictions (default: CCH_predictions)
- --gt_dir / GT_OUTPUT_DIR: directory for generated JSONs (default: CCH_ground_truth)
- --penalty / SMOOTH_PENALTY: smoothing switch penalty (default: 0.03)
- OTHER_THRESHOLD: map to Other if predicted score < threshold (default: 0.35)
- OTHER_FALLBACK_TO_STATEMENT: 1/0 fallback to STATEMENT when supported (default: 1)
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd


# ----------------- env + defaults -----------------
DEF_INPUT_DIR = os.getenv("INPUT_DIR", "CCH")
DEF_TRAIN_PDF = os.getenv("TRAIN_PDF_PATH", "content.pdf")
DEF_TRAIN_GT  = os.getenv("TRAIN_GT_PATH",  "23I_21PALCOACCT_V1_enhanced.json")
DEF_OUT_DIR   = os.getenv("OUTPUT_DIR", "CCH_predictions")
DEF_GT_DIR    = os.getenv("GT_OUTPUT_DIR", "CCH_ground_truth")
DEF_PENALTY   = float(os.getenv("SMOOTH_PENALTY", "0.03"))
OTHER_THRESHOLD = float(os.getenv("OTHER_THRESHOLD", "0.35"))
OTHER_FALLBACK_TO_STATEMENT = os.getenv("OTHER_FALLBACK_TO_STATEMENT", "1").strip() not in {"0", "false", "False"}


# ----------------- text/layout feature utils -----------------
STOP = set(
    """a an and are as at be but by for from has have if in into is it its of on or that the their this to was were will with within without your you they them he she we us our ours not nor than then which such page pages sequence number date name address city state zip employer identification taxpayer identification social security part section table column line""".split()
)


def tok(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t and t not in STOP and not t.isdigit()]


def header_text(page, frac=0.30):
    r = page.rect
    hrect = fitz.Rect(r.x0, r.y0, r.x1, r.y0 + r.height * frac)
    return page.get_text("text", clip=hrect) or ""


def largest_header_line(page, frac=0.30):
    d = page.get_text("dict")
    best = {"size": 0, "text": ""}
    header_ymax = page.rect.y0 + page.rect.height * frac
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            sizes = [s.get("size", 0) for s in l.get("spans", [])]
            texts = [s.get("text", "") for s in l.get("spans", [])]
            if not sizes or not texts:
                continue
            y0 = l.get("bbox", [0, 0, 0, 0])[1] if "bbox" in l else b.get("bbox", [0, 0, 0, 0])[1]
            if y0 > header_ymax:
                continue
            med = sorted(sizes)[len(sizes) // 2]
            if med > best["size"]:
                best = {"size": med, "text": " ".join(texts)}
    return re.sub(r"\s+", " ", best["text"]).strip()


def font_hist(page, bins=12):
    d = page.get_text("dict")
    sizes = []
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                sz = s.get("size", 0)
                if sz > 0:
                    sizes.append(sz)
    if not sizes:
        return [0.0] * bins
    mn, mx = min(sizes), max(sizes)
    if mx == mn:
        h = [0.0] * bins
        h[bins // 2] = 1.0
        return h
    h = [0] * bins
    for sz in sizes:
        idx = int((sz - mn) / (mx - mn) * (bins - 1))
        idx = max(0, min(bins - 1, idx))
        h[idx] += 1
    tot = sum(h) or 1
    return [x / tot for x in h]


def grid_occupancy(page, rows=8, cols=12):
    d = page.get_text("dict")
    rect = page.rect
    grid = [[0] * cols for _ in range(rows)]
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                bbox = s.get("bbox", None)
                if not bbox:
                    continue
                x0, y0, x1, y1 = bbox
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                r = int((cy - rect.y0) / rect.height * rows)
                c = int((cx - rect.x0) / rect.width * cols)
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r][c] = 1
    out = []
    for r in range(rows):
        out.extend(grid[r])
    s = sum(out) or 1
    return [v / s for v in out]


def draw_density(page, rows=4, cols=6):
    rect = page.rect
    grid = [[0] * cols for _ in range(rows)]
    for dr in page.get_drawings():
        bbox = dr.get("rect", None)
        if bbox:
            x0, y0, x1, y1 = bbox
        else:
            pts = []
            for p in dr.get("items", []):
                if p[0] == "l":
                    pts.append((p[1], p[2]))
            if not pts:
                continue
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        r = int((cy - rect.y0) / rect.height * rows)
        c = int((cx - rect.x0) / rect.width * cols)
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] += 1
    out = []
    for r in range(rows):
        out.extend(grid[r])
    tot = sum(out)
    return [v / tot if tot > 0 else 0 for v in out]


def cos_dict(a: dict, b: dict):
    dot = 0.0
    na = 0.0
    nb = 0.0
    for t, wa in a.items():
        na += wa * wa
        wb = b.get(t, 0.0)
        if wb:
            dot += wa * wb
    for wb in b.values():
        nb += wb * wb
    if na == 0 or nb == 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def cos_list(a: list, b: list):
    if (not a) or (not b) or len(a) != len(b):
        return 0.0
    a = np.array(a)
    b = np.array(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ----------------- training: build prototypes -----------------
def page_feature_bundles(page):
    full = page.get_text("text") or ""
    head = header_text(page, 0.30)
    tf = Counter(tok(full))
    for ht in tok(head):
        tf[ht] += 1
    title = Counter(tok(largest_header_line(page, 0.30)))
    font = font_hist(page, 12)
    grid = grid_occupancy(page, 8, 12)
    draw = draw_density(page, 4, 6)
    return tf, title, font, grid, draw


def build_prototypes(train_pdf_path: Path, train_gt_path: Path):
    with train_gt_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)
    page_gt = {p["page"]: p["label"] for p in ann["pages"]}
    labels = sorted({p["label"] for p in ann["pages"]})

    doc = fitz.open(str(train_pdf_path))
    text_centroid = {lbl: Counter() for lbl in labels}
    title_centroid = {lbl: Counter() for lbl in labels}
    font_proto = {lbl: np.zeros(12) for lbl in labels}
    grid_proto = {lbl: np.zeros(8 * 12) for lbl in labels}
    draw_proto = {lbl: np.zeros(4 * 6) for lbl in labels}
    counts = {lbl: 0 for lbl in labels}

    for i in range(doc.page_count):
        lbl = page_gt.get(i + 1, None)
        if not lbl:
            continue
        tf, title, font, grid, draw = page_feature_bundles(doc.load_page(i))
        text_centroid[lbl].update(tf)
        title_centroid[lbl].update(title)
        font_proto[lbl] += np.array(font)
        grid_proto[lbl] += np.array(grid)
        draw_proto[lbl] += np.array(draw)
        counts[lbl] += 1

    # normalize
    for lbl in labels:
        c = max(counts[lbl], 1)
        font_proto[lbl] = (font_proto[lbl] / c).tolist()
        grid_proto[lbl] = (grid_proto[lbl] / c).tolist()
        draw_proto[lbl] = (draw_proto[lbl] / c).tolist()
        tot = sum(text_centroid[lbl].values()) or 1
        text_centroid[lbl] = {k: v / tot for k, v in text_centroid[lbl].items()}
        ttot = sum(title_centroid[lbl].values()) or 1
        title_centroid[lbl] = {k: v / ttot for k, v in title_centroid[lbl].items()}

    return {
        "labels": labels,
        "text_centroid": text_centroid,
        "title_centroid": title_centroid,
        "font_proto": font_proto,
        "grid_proto": grid_proto,
        "draw_proto": draw_proto,
    }


# ----------------- inference helpers -----------------
def page_scores_with_protos(page, protos, alpha=0.65):
    tf, title, font, grid, draw = page_feature_bundles(page)
    labels = protos["labels"]
    scores = {}
    for lbl in labels:
        t = cos_dict(tf, protos["text_centroid"].get(lbl, {}))
        l = (
            0.35 * cos_dict(title, protos["title_centroid"].get(lbl, {}))
            + 0.35 * cos_list(font, protos["font_proto"].get(lbl, []))
            + 0.20 * cos_list(grid, protos["grid_proto"].get(lbl, []))
            + 0.10 * cos_list(draw, protos["draw_proto"].get(lbl, []))
        )
        scores[lbl] = alpha * t + (1.0 - alpha) * l
    return scores


def smooth_labels(score_mat, labels, switch_penalty=0.03):
    T, K = score_mat.shape
    dp = np.zeros((T, K), dtype=float)
    back = np.zeros((T, K), dtype=int)
    dp[0] = score_mat[0]
    back[0] = -1
    for t in range(1, T):
        for k in range(K):
            cont = dp[t - 1, k]
            switch = dp[t - 1] - switch_penalty
            if np.max(switch) > cont:
                dp[t, k] = score_mat[t, k] + np.max(switch)
                back[t, k] = int(np.argmax(dp[t - 1] - switch_penalty))
            else:
                dp[t, k] = score_mat[t, k] + cont
                back[t, k] = k
    y = np.zeros(T, dtype=int)
    y[T - 1] = int(np.argmax(dp[T - 1]))
    for t in range(T - 2, -1, -1):
        y[t] = back[t + 1, y[t + 1]]
    return [labels[i] for i in y]


def apply_other_rule(score_mat, labels, page_idx, pred_label, other_threshold, statement_label=None):
    if pred_label not in labels:
        return "Other"
    pred_idx = labels.index(pred_label)
    pred_score = float(score_mat[page_idx, pred_idx])
    if pred_score < other_threshold:
        if statement_label is not None:
            st_idx = labels.index(statement_label)
            st_score = float(score_mat[page_idx, st_idx])
            if st_score >= other_threshold:
                return statement_label
        return "Other"
    return pred_label


def family_from_label(label: str) -> str:
    if not label:
        return "Other"
    if label.lower() == "other":
        return "Other"
    # prefer common 4-digit form numbers if present
    m = re.search(r"(?:Form_)?(\d{3,4})", label)
    if m:
        return m.group(1)
    # strip common suffixes to get base family
    s = re.sub(r"_STMT(?:_PG_\d+)?$", "", label, flags=re.IGNORECASE)
    s = re.sub(r"_(?:PG|Pg)_?\d+.*$", "", s)
    return s or label


def write_gt_json(pdf_path: Path, labels_seq, gt_dir: Path, config_name: str = "forms_config_enhanced") -> Path:
    gt_dir.mkdir(parents=True, exist_ok=True)
    base = pdf_path.stem
    out_path = gt_dir / f"{base}_enhanced.json"
    pages = []
    for i, lbl in enumerate(labels_seq, start=1):
        pages.append({
            "page": i,
            "family": family_from_label(lbl),
            "label": lbl,
        })
    obj = {
        "document": pdf_path.name,
        "config": config_name,
        "total_pages": len(labels_seq),
        "pages": pages,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return out_path


def classify_pdf(pdf_path: Path, protos, other_threshold=OTHER_THRESHOLD, penalty=DEF_PENALTY, out_dir: Path = None, gt_dir: Path = None):
    labels = protos["labels"]
    other_label = next((l for l in labels if str(l).lower() == "other"), "Other")
    statement_label = next((l for l in labels if str(l).lower() in {"statement", "statemet"}), None)
    if not OTHER_FALLBACK_TO_STATEMENT:
        statement_label = None

    doc = fitz.open(str(pdf_path))
    P = doc.page_count
    score_mat = np.zeros((P, len(labels)), dtype=float)
    for i in range(P):
        s = page_scores_with_protos(doc.load_page(i), protos, alpha=0.65)
        for j, lbl in enumerate(labels):
            score_mat[i, j] = s[lbl]

    # smoothing + Other mapping
    raw_seq = smooth_labels(score_mat, labels, switch_penalty=penalty)
    final_seq = [apply_other_rule(score_mat, labels, i, raw_seq[i], other_threshold, statement_label) for i in range(P)]

    # write predictions
    base = pdf_path.stem
    out = pd.DataFrame({
        "page": np.arange(1, P + 1),
        "predicted_label": final_seq,
    })
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{base}_predictions.csv"
    else:
        out_path = pdf_path.with_suffix("")
        out_path = out_path.with_name(f"{out_path.name}_predictions.csv")
    out.to_csv(out_path, index=False)

    gt_path = None
    if gt_dir is not None:
        gt_path = write_gt_json(pdf_path, final_seq, gt_dir)
    return out_path, gt_path


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Classify unlabeled PDFs using prototypes from a labeled training PDF")
    ap.add_argument("--input_dir", default=DEF_INPUT_DIR, help="Folder containing PDFs to classify (default: CCH)")
    ap.add_argument("--train_pdf", default=DEF_TRAIN_PDF, help="Training PDF path (default: content.pdf)")
    ap.add_argument("--train_gt", default=DEF_TRAIN_GT, help="Training ground-truth JSON path")
    ap.add_argument("--out_dir", default=DEF_OUT_DIR, help="Folder to write predictions (default: CCH_predictions)")
    ap.add_argument("--gt_dir", default=DEF_GT_DIR, help="Folder to write generated ground-truth JSONs (default: CCH_ground_truth)")
    ap.add_argument("--penalty", type=float, default=DEF_PENALTY, help="Smoothing switch penalty (default: 0.03)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    train_pdf = Path(args.train_pdf)
    train_gt = Path(args.train_gt)
    out_dir = Path(args.out_dir) if args.out_dir else None
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    penalty = float(args.penalty)

    if not train_pdf.exists():
        raise FileNotFoundError(f"Training PDF not found: {train_pdf}")
    if not train_gt.exists():
        raise FileNotFoundError(f"Training GT JSON not found: {train_gt}")
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    protos = build_prototypes(train_pdf, train_gt)
    labels = protos["labels"]
    print(f"Loaded {len(labels)} labels from training GT.")

    # enumerate PDFs in input_dir
    pdfs = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        return

    written = []
    for pdf in pdfs:
        try:
            out_path, gt_path = classify_pdf(pdf, protos, other_threshold=OTHER_THRESHOLD, penalty=penalty, out_dir=out_dir, gt_dir=gt_dir)
            written.append((pdf.name, str(out_path), str(gt_path) if gt_path else ""))
            msg = f"Classified {pdf.name} -> {out_path}"
            if gt_path:
                msg += f"; GT -> {gt_path}"
            print(msg)
        except Exception as e:
            print(f"Error classifying {pdf}: {e}")

    print("\nSummary:")
    for name, outp, gtp in written:
        extra = f"; GT: {gtp}" if gtp else ""
        print(f"  - {name} -> {outp}{extra}")


if __name__ == "__main__":
    main()
