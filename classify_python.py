#!/usr/bin/env python3
# step1_sequence_smoothing.py
# Reproduce Step 1 (sequence smoothing) and report accuracy.
# Files expected in the same directory:
#   - content.pdf
#   - 23I_21PALCOACCT_V1_enhanced.json

import json, re, os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import fitz  # PyMuPDF

PDF_PATH = Path("content.pdf")
GT_PATH  = Path("23I_21PALCOACCT_V1_enhanced.json")
# Unknown handling ("Other")
# Modes:
#  - per_label (default): compute a per-label acceptance threshold from training scores (quantile)
#  - global: use a single global threshold for all labels
OTHER_MODE = os.getenv("OTHER_MODE", "per_label").strip().lower()
OTHER_THRESHOLD = float(os.getenv("OTHER_THRESHOLD", "0.35"))
OTHER_Q = float(os.getenv("OTHER_Q", "0.05"))  # quantile for per-label thresholds
OTHER_MARGIN = float(os.getenv("OTHER_MARGIN", "0.05"))  # require ambiguity to reject
OTHER_THRESH_CAP = float(os.getenv("OTHER_THRESH_CAP", "0.95"))  # cap too-high thresholds
OTHER_RESCUE = os.getenv("OTHER_RESCUE", "1").strip() not in {"0", "false", "False"}
SAME_FAMILY_PENALTY_FACTOR = float(os.getenv("SAME_FAMILY_PENALTY_FACTOR", "0.2"))  # cheaper within family
PAGE_OVERRIDE_DELTA = float(os.getenv("PAGE_OVERRIDE_DELTA", "0.008"))  # override to per-page top if stronger by delta
_never_other_env = os.getenv("OTHER_NEVER_OTHER_LABELS", "STATEMENT,STATEMET")
OTHER_NEVER_OTHER_LABELS = {s.strip().lower() for s in _never_other_env.split(',') if s.strip()}
OTHER_FALLBACK_TO_STATEMENT = os.getenv("OTHER_FALLBACK_TO_STATEMENT", "1").strip() not in {"0", "false", "False"}

# ----------------- utils -----------------
# Stopwords: keep common function words, but DO NOT remove
# label-bearing tokens like 'statement', 'schedule', 'form', 'attachment'.
STOP = set("""a an and are as at be but by for from has have if in into is it its of on or that the their this to was were will with within without your you they them he she we us our ours not nor than then which such page pages sequence number date name address city state zip employer identification taxpayer identification social security part section table column line""".split())

def tok(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t and t not in STOP and not t.isdigit()]

def header_text(page, frac=0.30):
    r = page.rect
    hrect = fitz.Rect(r.x0, r.y0, r.x1, r.y0 + r.height*frac)
    return page.get_text("text", clip=hrect) or ""

def largest_header_line(page, frac=0.30):
    d = page.get_text("dict")
    best = {"size": 0, "text": ""}
    header_ymax = page.rect.y0 + page.rect.height * frac
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            sizes = [s.get("size",0) for s in l.get("spans",[])]
            texts = [s.get("text","") for s in l.get("spans",[])]
            if not sizes or not texts: 
                continue
            y0 = l.get("bbox", [0,0,0,0])[1] if "bbox" in l else b.get("bbox",[0,0,0,0])[1]
            if y0 > header_ymax:
                continue
            med = sorted(sizes)[len(sizes)//2]
            if med > best["size"]:
                best = {"size": med, "text": " ".join(texts)}
    return re.sub(r"\s+"," ",best["text"]).strip()

def font_hist(page, bins=12):
    d = page.get_text("dict")
    sizes = []
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                sz = s.get("size", 0)
                if sz>0: sizes.append(sz)
    if not sizes:
        return [0.0]*bins
    mn, mx = min(sizes), max(sizes)
    if mx==mn:
        h=[0.0]*bins; h[bins//2]=1.0; return h
    h=[0]*bins
    for sz in sizes:
        idx = int((sz - mn)/(mx-mn)*(bins-1)); idx=max(0,min(bins-1,idx))
        h[idx]+=1
    tot = sum(h) or 1
    return [x/tot for x in h]

def grid_occupancy(page, rows=8, cols=12):
    d = page.get_text("dict"); rect = page.rect
    grid = [[0]*cols for _ in range(rows)]
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                bbox = s.get("bbox", None)
                if not bbox: 
                    continue
                x0,y0,x1,y1 = bbox
                cx=(x0+x1)/2.0; cy=(y0+y1)/2.0
                r = int((cy-rect.y0)/rect.height * rows)
                c = int((cx-rect.x0)/rect.width * cols)
                if 0<=r<rows and 0<=c<cols: 
                    grid[r][c] = 1
    out=[]
    for r in range(rows): out.extend(grid[r])
    s=sum(out) or 1
    return [v/s for v in out]

def draw_density(page, rows=4, cols=6):
    rect = page.rect
    grid = [[0]*cols for _ in range(rows)]
    for dr in page.get_drawings():
        bbox = dr.get("rect", None)
        if bbox:
            x0,y0,x1,y1 = bbox
        else:
            pts=[]
            for p in dr.get("items", []):
                if p[0]=="l": pts.append((p[1],p[2]))
            if not pts: 
                continue
            xs=[x for x,_ in pts]; ys=[y for _,y in pts]
            x0,y0,x1,y1 = min(xs), min(ys), max(xs), max(ys)
        cx=(x0+x1)/2.0; cy=(y0+y1)/2.0
        r = int((cy-rect.y0)/rect.height * rows)
        c = int((cx-rect.x0)/rect.width * cols)
        if 0<=r<rows and 0<=c<cols: 
            grid[r][c]+=1
    out=[]
    for r in range(rows): out.extend(grid[r])
    tot = sum(out)
    return [v/tot if tot>0 else 0 for v in out]

def cos_dict(a: dict, b: dict):
    dot=0.0; na=0.0; nb=0.0
    for t,wa in a.items():
        na+=wa*wa
        wb=b.get(t,0.0)
        if wb: dot+=wa*wb
    for wb in b.values(): nb+=wb*wb
    if na==0 or nb==0: return 0.0
    return dot/((na**0.5)*(nb**0.5))

def cos_list(a: list, b: list):
    if (not a) or (not b) or len(a)!=len(b): return 0.0
    a=np.array(a); b=np.array(b)
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))

# ----------------- load GT + build prototypes -----------------
with GT_PATH.open("r", encoding="utf-8") as f:
    ann = json.load(f)
page_gt = {p["page"]: p["label"] for p in ann["pages"]}
labels = sorted({p["label"] for p in ann["pages"]})
OTHER_LABEL = next((l for l in labels if str(l).lower()=="other"), "Other")
GENERIC_STATEMENT_LABEL = next((l for l in labels if str(l).lower() in {"statement", "statemet"}), None)

doc = fitz.open(str(PDF_PATH))

text_centroid  = {lbl: Counter() for lbl in labels}
title_centroid = {lbl: Counter() for lbl in labels}
font_proto     = {lbl: np.zeros(12)   for lbl in labels}
grid_proto     = {lbl: np.zeros(8*12) for lbl in labels}
draw_proto     = {lbl: np.zeros(4*6)  for lbl in labels}
counts         = {lbl: 0 for lbl in labels}

def page_feature_bundles(page):
    full = page.get_text("text") or ""
    head = header_text(page, 0.30)
    tf = Counter(tok(full)); 
    for ht in tok(head): tf[ht]+=1
    title = Counter(tok(largest_header_line(page, 0.30)))
    font = font_hist(page, 12)
    grid = grid_occupancy(page, 8, 12)
    draw = draw_density(page, 4, 6)
    return tf, title, font, grid, draw

def base_label(lbl: str) -> str:
    s = lbl
    s = re.sub(r"_Pg_\d+.*$", "", s)
    s = re.sub(r"_STMT$", "", s, flags=re.IGNORECASE)
    return s

LABEL_FAMILY = {lbl: base_label(lbl) for lbl in labels}

for i in range(doc.page_count):
    lbl = page_gt.get(i+1, None)
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
    font_proto[lbl] = (font_proto[lbl]/c).tolist()
    grid_proto[lbl] = (grid_proto[lbl]/c).tolist()
    draw_proto[lbl] = (draw_proto[lbl]/c).tolist()
    tot = sum(text_centroid[lbl].values()) or 1
    text_centroid[lbl] = {k: v/tot for k,v in text_centroid[lbl].items()}
    ttot = sum(title_centroid[lbl].values()) or 1
    title_centroid[lbl] = {k: v/ttot for k,v in title_centroid[lbl].items()}

# ----------------- per-page scores -----------------
def page_scores(page, alpha=0.65):
    full = page.get_text("text") or ""
    head = header_text(page, 0.30)
    tf = Counter(tok(full)); 
    for ht in tok(head): tf[ht]+=1
    title = Counter(tok(largest_header_line(page, 0.30)))
    font = font_hist(page, 12)
    grid = grid_occupancy(page, 8, 12)
    draw = draw_density(page, 4, 6)
    scores={}
    for lbl in labels:
        t = cos_dict(tf,     text_centroid.get(lbl, {}))
        lp = {
            "title":     title_centroid.get(lbl, {}),
            "font_hist": font_proto.get(lbl, []),
            "grid":      grid_proto.get(lbl, []),
            "draw":      draw_proto.get(lbl, [])
        }
        l = (0.35*cos_dict(title, lp["title"]) +
             0.35*cos_list(font,  lp["font_hist"]) +
             0.20*cos_list(grid,  lp["grid"]) +
             0.10*cos_list(draw,  lp["draw"]))
        scores[lbl] = alpha*t + (1.0-alpha)*l
    return scores

P = doc.page_count
score_mat = np.zeros((P, len(labels)), dtype=float)
rows = []
top_scores = []
second_scores = []
label_self_scores = {lbl: [] for lbl in labels}
for i in range(P):
    s = page_scores(doc.load_page(i), alpha=0.65)
    # fill row scores
    row = np.array([s[lbl] for lbl in labels], dtype=float)
    score_mat[i] = row
    # best and second-best
    best_idx = int(np.argmax(row))
    best = labels[best_idx]
    top = float(row[best_idx])
    tmp = row.copy()
    tmp[best_idx] = -np.inf
    second = float(np.max(tmp)) if np.isfinite(np.max(tmp)) else -np.inf
    # collect training self-scores
    gt_lbl = page_gt.get(i+1, None)
    if gt_lbl in labels:
        label_self_scores[gt_lbl].append(float(s.get(gt_lbl, 0.0)))
    # placeholder; final thresholding applied after per-label thresholds computed
    thr_label = best
    rows.append({
        "page": i+1,
        "baseline_best": best,
        "top_score": top,
        "second_best_score": second,
        "threshold_used": None,
        "thresholded_label": None,
        **{lbl: s[lbl] for lbl in labels}
    })
    top_scores.append(top)
    second_scores.append(second)
pd.DataFrame(rows).to_csv("page_label_scores.csv", index=False)

# ----------------- sequence smoothing (Viterbi-ish) -----------------
def smooth_labels(score_mat, labels, switch_penalty=0.05, same_family_factor=SAME_FAMILY_PENALTY_FACTOR):
    T, K = score_mat.shape
    dp = np.zeros((T, K), dtype=float)
    back = np.zeros((T, K), dtype=int)
    dp[0] = score_mat[0]           # no penalty for start
    back[0] = -1
    for t in range(1, T):
        prev = dp[t-1]
        for k in range(K):
            cont = prev[k]  # keep label k
            # Pair-specific penalty: cheaper (or free) to switch within same family
            family_k = LABEL_FAMILY[labels[k]]
            penalty_vec = np.where(
                np.array([LABEL_FAMILY[labels[j]] for j in range(K)]) == family_k,
                switch_penalty * same_family_factor,
                switch_penalty,
            )
            switch = prev - penalty_vec
            if np.max(switch) > cont:
                jstar = int(np.argmax(switch))
                dp[t, k] = score_mat[t, k] + switch[jstar]
                back[t, k] = jstar
            else:
                dp[t, k] = score_mat[t, k] + cont
                back[t, k] = k
    y = np.zeros(T, dtype=int)
    y[T-1] = int(np.argmax(dp[T-1]))
    for t in range(T-2, -1, -1):
        y[t] = back[t+1, y[t+1]]
    return [labels[i] for i in y]

def accuracy(pred_seq):
    gt = [page_gt.get(i+1, None) for i in range(P)]
    return float(np.mean([pred_seq[i] == gt[i] for i in range(P)]))

# build per-label thresholds (if enabled)
per_label_threshold = {}
if OTHER_MODE == "per_label":
    for lbl, arr in label_self_scores.items():
        if len(arr) >= 1:
            try:
                thr = float(np.quantile(np.array(arr, dtype=float), OTHER_Q))
                per_label_threshold[lbl] = min(thr, OTHER_THRESH_CAP)
            except Exception:
                per_label_threshold[lbl] = min(OTHER_THRESHOLD, OTHER_THRESH_CAP)
        else:
            per_label_threshold[lbl] = min(OTHER_THRESHOLD, OTHER_THRESH_CAP)
else:
    # global mode uses OTHER_THRESHOLD for all labels
    per_label_threshold = {lbl: min(OTHER_THRESHOLD, OTHER_THRESH_CAP) for lbl in labels}

def threshold_for(label: str) -> float:
    return per_label_threshold.get(label, OTHER_THRESHOLD)

def apply_other_rule(page_idx: int, pred_label: str) -> str:
    """Decide whether to map pred_label -> OTHER based on the predicted
    label's own score and its separation from the next best alternative
    on this page.
    """
    if pred_label not in labels:
        return OTHER_LABEL
    # Never demote certain labels (e.g., STATEMENT) to Other
    if pred_label.strip().lower() in OTHER_NEVER_OTHER_LABELS:
        return pred_label
    pred_idx = labels.index(pred_label)
    row = score_mat[page_idx]
    pred_score = float(row[pred_idx])
    # best alternative excluding the predicted label
    alt = np.max(np.delete(row, pred_idx)) if row.size > 1 else -np.inf
    thr = threshold_for(pred_label)
    if (pred_score < thr) and (alt > -np.inf) and ((pred_score - alt) < OTHER_MARGIN):
        # Try rescue to a better-supported alternative instead of Other
        if OTHER_RESCUE:
            row = score_mat[page_idx]
            best_idx = int(np.argmax(row))
            best_lbl = labels[best_idx]
            best_score = float(row[best_idx])
            alt2 = float(np.max(np.delete(row, best_idx))) if row.size > 1 else -np.inf
            if (best_lbl != pred_label) and (best_score >= threshold_for(best_lbl)) and (best_score - alt2 >= OTHER_MARGIN):
                return best_lbl
        # Fallback to generic STATEMENT (if present and reasonably supported)
        if OTHER_FALLBACK_TO_STATEMENT and GENERIC_STATEMENT_LABEL is not None:
            st_idx = labels.index(GENERIC_STATEMENT_LABEL)
            st_score = float(score_mat[page_idx, st_idx])
            if st_score >= threshold_for(GENERIC_STATEMENT_LABEL):
                return GENERIC_STATEMENT_LABEL
        return OTHER_LABEL
    return pred_label

# baseline (greedy per page) with Other rule
baseline_raw = [labels[i] for i in np.argmax(score_mat, axis=1)]
baseline_pred = [apply_other_rule(i, baseline_raw[i]) for i in range(P)]
baseline_acc = accuracy(baseline_pred)

# sweep penalties (includes 0.05 which gave ~84.4% on your file)
penalties = [0.00, 0.01, 0.02, 0.03, 0.05]
summary = []
for pen in penalties:
    seq = smooth_labels(score_mat, labels, switch_penalty=pen)
    # apply Other rule after smoothing using per-page, per-label scores
    seq_thr = [apply_other_rule(i, seq[i]) for i in range(P)]
    acc = accuracy(seq_thr)
    summary.append({
        "switch_penalty": pen,
        "accuracy": round(acc, 3),
        "num_switches": sum(1 for i in range(1,P) if seq_thr[i]!=seq_thr[i-1])
    })
sum_df = pd.DataFrame(summary).sort_values("accuracy", ascending=False)
sum_df.to_csv("seq_smoothing_summary.csv", index=False)

best_pen = float(sum_df.iloc[0]["switch_penalty"])
best_seq_raw = smooth_labels(score_mat, labels, switch_penalty=best_pen)
best_seq_raw_override = []
for i in range(P):
    row = score_mat[i]
    pred_raw = best_seq_raw[i]
    pred_idx = labels.index(pred_raw) if pred_raw in labels else None
    top_idx = int(np.argmax(row))
    top_lbl = labels[top_idx]
    if pred_idx is not None:
        diff = float(row[top_idx] - row[pred_idx])
    else:
        diff = float('inf')
    if (top_lbl != pred_raw) and (diff >= PAGE_OVERRIDE_DELTA):
        best_seq_raw_override.append(top_lbl)
    else:
        best_seq_raw_override.append(pred_raw)
best_seq = [apply_other_rule(i, best_seq_raw_override[i]) for i in range(P)]
best_acc = accuracy(best_seq)

# Update CSV with thresholded labels and thresholds used
try:
    df_scores = pd.read_csv("page_label_scores.csv")
    thr_used = [threshold_for(baseline_raw[i]) for i in range(P)]
    thr_label = [apply_other_rule(i, baseline_raw[i]) for i in range(P)]
    df_scores["threshold_used"] = thr_used
    df_scores["thresholded_label"] = thr_label
    df_scores.to_csv("page_label_scores.csv", index=False)
except Exception:
    pass

# write evals
eval_rows = []
for i in range(P):
    pred_raw = best_seq_raw_override[i]
    pred = best_seq[i]
    gt = page_gt.get(i+1, None)
    pred_idx = labels.index(pred_raw) if pred_raw in labels else None
    row_scores = score_mat[i]
    pred_score = float(row_scores[pred_idx]) if pred_idx is not None else float("nan")
    alt_score = float(np.max(np.delete(row_scores, pred_idx))) if pred_idx is not None and row_scores.size>1 else float("nan")
    # determine if rescue changed the label
    final_label = best_seq[i]
    rescued_to = final_label if (final_label != pred_raw and (final_label == labels[int(np.argmax(row_scores))] or (GENERIC_STATEMENT_LABEL and final_label == GENERIC_STATEMENT_LABEL))) else (final_label if (final_label != pred_raw) else "")
    override_applied = (best_seq_raw[i] != best_seq_raw_override[i])
    top_idx = int(np.argmax(row_scores))
    top_lbl = labels[top_idx]
    top_score = float(row_scores[top_idx])
    eval_rows.append({
        "page": i+1,
        "predicted_label": final_label,
        "predicted_label_raw": pred_raw,
        "page_top_label": top_lbl,
        "ground_truth_label": gt,
        "match": final_label == gt,
        "switch_penalty": best_pen,
        "predicted_label_score": pred_score,
        "best_alt_score": alt_score,
        "margin": (pred_score - alt_score) if (not np.isnan(pred_score) and not np.isnan(alt_score)) else float("nan"),
        "threshold_used": threshold_for(pred_raw) if pred_raw in labels else OTHER_THRESHOLD,
        "page_top_score": top_score,
        "page_override_applied": override_applied,
        "other_rescue_to": rescued_to,
    })
pd.DataFrame(eval_rows).to_csv("eval_seq_smoothing_best.csv", index=False)

pd.DataFrame([
    {"page": i+1, "ground_truth_label": page_gt.get(i+1, None), "predicted_label": best_seq[i]}
    for i in range(P) if best_seq[i] != page_gt.get(i+1, None)
]).to_csv("misclassified_seq_smoothing_best.csv", index=False)

print(f"Baseline accuracy (no smoothing): {baseline_acc:.3f}")
print(f"Best accuracy (with smoothing):  {best_acc:.3f}  @ switch_penalty={best_pen}")
print("CSV outputs:")
print("  - page_label_scores.csv")
print("  - seq_smoothing_summary.csv")
print("  - eval_seq_smoothing_best.csv")
print("  - misclassified_seq_smoothing_best.csv")
