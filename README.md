**Overview**
- This project classifies each page of a PDF into section labels using a combination of text and layout features, then applies sequence smoothing to reduce spurious label flips across pages.
- The main script is `classify_python.py` which reads a PDF and a ground‑truth JSON, learns simple per‑label prototypes, scores each page, and writes evaluation CSVs.

**Inputs**
- `content.pdf` — The PDF whose pages are to be classified.
- `23I_21PALCOACCT_V1_enhanced.json` — Ground truth with page‑level labels used to learn label prototypes and compute accuracy.
- `requirements.txt` — Python dependencies (`pymupdf`, `pandas`, `numpy`).

**Ground Truth Format**
- Expected JSON structure (minimum):
  - Top‑level key `pages` containing a list of objects, each with:
    - `page` (1‑based page index)
    - `label` (string label for that page)
- Example:
  - `{ "pages": [ { "page": 1, "label": "cover" }, { "page": 2, "label": "toc" } ] }`

**How Classification Works**
- Feature extraction per page (via PyMuPDF):
  - Text tokens: lowercased, punctuation stripped, stopwords removed; tokens from full page text with a slight boost for header area.
  - Title tokens: tokens from the largest text line within the top header region.
  - Font histogram: normalized histogram of detected font sizes.
  - Grid occupancy: an 8×12 grid marking where text spans appear (layout density).
  - Drawing density: a 4×6 grid counting vector drawings/lines (e.g., tables, boxes).
- Label prototypes (learned from GT pages):
  - Text centroid: normalized token frequencies per label.
  - Title centroid: normalized title-token frequencies per label.
  - Average font histogram, grid occupancy, drawing density per label.
- Per‑page label scoring:
  - Compute cosine similarity between the page features and each label’s prototypes.
  - Combine similarities:
    - `alpha * text_similarity + (1 - alpha) * layout_similarity`
    - Layout similarity weights: title 0.35, font 0.35, grid 0.20, draw 0.10.
  - The label with the highest score is the page’s baseline prediction.
- Sequence smoothing:
  - A simple Viterbi‑like dynamic program penalizes label changes between adjacent pages (`switch_penalty`).
  - The script sweeps several penalties and picks the best based on accuracy versus ground truth.

**Running**
- Python 3.9+ recommended.
- Optional: create and activate a virtual environment
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
- Install dependencies
  - `pip install -r requirements.txt`
- Place `content.pdf` and `23I_21PALCOACCT_V1_enhanced.json` in the project root (or update paths in `classify_python.py`: `PDF_PATH`, `GT_PATH`).
- Run
  - `python classify_python.py`

**Outputs**
- `page_label_scores.csv` — Per‑page, per‑label scores with `baseline_best`.
- `seq_smoothing_summary.csv` — Accuracy and number of switches for each tested `switch_penalty`.
- `eval_seq_smoothing_best.csv` — Final per‑page predictions using the selected best penalty: `page`, `predicted_label`, `ground_truth_label`, `match`.
- `misclassified_seq_smoothing_best.csv` — Only the pages where prediction ≠ ground truth.
- Console prints baseline accuracy (no smoothing) and best smoothed accuracy with the chosen penalty.

**Unknown / Other Handling**
- Intent: assign `Other` only when the predicted label’s score is too low.
- Rule: if the predicted label’s score < `OTHER_THRESHOLD` (default `0.35`), map to `Other`.
- STATEMENT fallback: if a generic `STATEMENT` label exists and its score ≥ `OTHER_THRESHOLD`, prefer `STATEMENT` instead of `Other`.
- Configuration (env vars):
  - `OTHER_THRESHOLD` — default `0.35`.
  - `OTHER_FALLBACK_TO_STATEMENT` — `1`/`0` to enable/disable fallback (default enabled).
- If your GT already includes a label named `Other` (any case), the script uses that exact label string; otherwise it uses `Other`.

**Tuning and Customization**
- File paths: edit `PDF_PATH` and `GT_PATH` at the top of `classify_python.py`.
- Text processing: adjust the `STOP` word list or tokenization in `tok()`.
  - Note: tokens like `statement`, `schedule`, `form`, and `attachment` are intentionally NOT stopped because they are often label-bearing and help classification.
- Header region: change the fraction height used in `header_text()` and `largest_header_line()` (default 0.30 of page height).
- Feature resolutions: modify bins/grids in `font_hist()`, `grid_occupancy()`, and `draw_density()`.
- Scoring weights: tweak `alpha` (text vs layout) in `page_scores()` and the layout sub‑weights (title/font/grid/draw).
- Smoothing: update the candidate `penalties` list and default `switch_penalty` in `smooth_labels()` sweep.
- Other threshold: set `OTHER_THRESHOLD` env var or change the default near the top of `classify_python.py`.

 

**Troubleshooting**
- Import error `fitz`: ensure `pymupdf` is installed and Python version is compatible.
- Incorrect paths: verify `content.pdf` and the JSON are in the working directory or adjust the constants.
- Empty or unusual PDFs: some features rely on extractable text/spans/drawings; scanned PDFs without OCR will degrade text features.

**Notes**
- The script both learns from and evaluates on the same document; it is designed to demonstrate the method and quantify page‑wise smoothing gains. For generalization to other documents, provide a separate training set to build prototypes, then apply to new PDFs without including them in prototype construction.
