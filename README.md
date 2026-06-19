# DataScienceProject — Arabic Document CV/OCR Pipeline for Tunisian ID Documents

> An end-to-end computer-vision + OCR pipeline that **classifies, deskews, cleans, and reads Arabic fields** from real-world Tunisian official documents — the CIN national ID card (recto + verso) and the *carte grise* vehicle registration — then serializes the extracted fields to JSON and **auto-fills a contract template**.

![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![EasyOCR](https://img.shields.io/badge/EasyOCR-00897B?style=for-the-badge&logo=tesseract&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## Overview

**The problem.** Tunisian administrative documents carry their key information in **Arabic** — a cursive, right-to-left script that off-the-shelf OCR handles far worse than Latin text. In practice these documents arrive as phone photos: **skewed, glare-affected, unevenly lit, low-resolution**, and cluttered with printed labels next to the fields you actually want. Reading those fields reliably enough to populate a paper contract is impossible without a full image-preparation chain *before* any OCR runs.

**The pipeline.** This project is a **6-stage** vision pipeline, each stage prototyped in its own notebook and then assembled end-to-end in `integration_notebook.ipynb`:

```
   [1] CLASSIFY          [2] ORIENT / DESKEW       [3] LOCALIZE & CLEAN
   CNN: CIN vs.     →    projection-profile    →   face anchor + contour crop
   carte grise           score sweep + Haar         + K-channel threshold
                                                     + glare inpaint + label mask
                                                              │
                                                              ▼
   [6] CONTRACT AUTOFILL  ←  [5] SERIALIZE   ←   [4] ARABIC OCR + PARSE
   Pillow draw + bidi          fields → JSON       OCR.space → ArabicOCR → EasyOCR
                                                   + reshape/bidi + regex fields
```

**Two document types, six output fields.** The pipeline targets **CIN recto** (`cin`, `nom`), **CIN verso** (`profession`, `adresse`), and **carte grise** (`serie_type`, `dpmc`), all of which are written into the final contract.

---

## Architecture & rationale — why each stage exists

Each stage is justified by a concrete failure mode of the stage *after* it. The whole pipeline is "front-load every transformation that OCR is sensitive to, so the OCR call sees the cleanest possible crop."

### Stage 1 — Document-type classification (CNN)
A small **convolutional network** decides *carte d'identité* vs. *carte grise* before any document-specific logic runs, because the two types need completely different localization (a CIN is anchored on the face; a carte grise is anchored on a red flag).

- **Architecture** (`data_augmentation.ipynb`): input `100×100×3` → `Conv2D(16, 3×3, same, ReLU)` → `MaxPool` → `Conv2D(32)` → `MaxPool` → `Conv2D(64)` → `MaxPool` → `Dropout(0.2)` → `Flatten` → `Dense(128, ReLU)` → `Dense(2)`. Compiled with **Adam** + **BinaryCrossentropy**, trained **30 epochs** on a `train_test_split` from scikit-learn.
- **Why heavy augmentation:** the dataset is small and locally collected, so the model is fronted by a `keras.Sequential` augmentation stack — **~9 `RandomZoom`** levels, **~9 `RandomContrast`** levels, and `RandomFlip("horizontal_and_vertical")` — to manufacture invariance to the scale, contrast, and orientation noise of phone photos.

### Stage 2 — Orientation / deskew
Arabic OCR is **orientation-sensitive**: a few degrees of skew collapses recognition of a cursive script. Two complementary methods are implemented:

- **Projection-profile score sweep** (`correct_orientation` in the integration notebook): Otsu-binarize, then rotate the image across **−90°…+90° in 0.1° steps**, scoring each angle by the **squared difference of the row-sum histogram** (`Σ(hist[i+1]−hist[i])²`). The angle with the sharpest horizontal banding wins — that is the angle at which text lines are horizontal.
- **Geometry-based fallbacks**: `Orientation.py` derives the skew from `minAreaRect` on the Otsu mask; `orientation_quelque_soit_angle.ipynb` estimates the dominant axis with **PCA (`cv2.PCACompute2`)** on the contour point cloud for arbitrary input angles; `rotation.py` is a brute-force search that rotates across angles and **keeps the orientation that yields the longest OCR string** (a recognition-driven tiebreaker).

### Stage 3 — Localization & cleaning
The OCR must see *only the fields*, not the surrounding card art and labels.

- **CIN recto** — a **Haar cascade** (`haarcascade`) detects the face; the field block is then cropped *relative to the face box and the OCR text-overlay geometry* (max `Top`/`Left` of detected words), so the crop adapts to each photo instead of using fixed coordinates.
- **Carte grise** — a **dual-range red HSV mask** (Hue `0–10` and `160–179`) finds the small red flag; candidate contours are filtered by aspect ratio and size, and a **K-means dominant-color check** (`n_colors=7`, red channel ≥ 1.8× green) confirms the flag before using it as an anchor to crop the registration block.
- **Glare / flash removal** (`projet.py`) — bright specular regions are masked with a `220` threshold and reconstructed with **`cv2.inpaint` (TELEA)**; printed labels are inpainted out (`INPAINT_NS`) after an EasyOCR pass locates them (`noNeeded_text_mask.ipynb`).
- **Adaptive brightness** — mean luminance is measured with `PIL.ImageStat` and the V-channel is boosted on a **16-step ladder**, so dark photos are lifted without blowing out already-bright ones.
- **K-channel binarization** — fields are isolated via the printing-style **K (key/black) channel** (`1 − max(R,G,B)`) thresholded at ~140–150, which separates dark Arabic ink from a colored card background better than a plain grayscale threshold.

### Stage 4 — Arabic OCR + field parsing
OCR runs on the cleaned crop through a **layered fallback chain**, because no single Arabic OCR engine is reliable on official-document imagery:

1. **OCR.space API** (`language=ara`, `isOverlayRequired=true`) — primary; its **`TextOverlay` line/word geometry** is reused upstream to drive the crop.
2. **ArabicOCR** (`ArabicOcr.arabicocr.arabic_ocr`) — fallback when the API returns no `ParsedResults`.
3. **EasyOCR** (`Reader(['ar','en'])`) — used for the carte grise, where a **regex date pattern** extracts the `DPMC` and an **alphanumeric-length filter** (`>9` chars, mixed letters+digits) extracts the serial/type number.

Arabic text is then **reshaped** (`arabic_reshaper`) so letters take their correct contextual forms, and **reordered** with `python-bidi` (`get_display`) so the right-to-left string renders correctly when drawn left-to-right. `python-Levenshtein` supports fuzzy field matching/cleanup.

### Stage 5 — Serialization
The six fields are assembled into a dict and written as **UTF-8 JSON** (`json.dumps(..., indent=4)`) — a clean, inspectable hand-off boundary between the vision pipeline and the document-generation step.

### Stage 6 — Contract auto-fill
The JSON is rendered onto a **contract image template** with Pillow (`ImageDraw.text` at fixed field coordinates, `arial.ttf`). Arabic values are passed through `bidi.get_display` again so they appear correctly on the rendered contract.

---

## Why this is hard (and under-served)

This is not "wrap Tesseract in a loop." The difficulty is structural:

- **Arabic is the hard case for OCR.** It is cursive (letters change shape by position), right-to-left, and diacritic-bearing. Mainstream OCR tooling is tuned for Latin scripts; Arabic accuracy degrades sharply, and **correct display requires an explicit reshape + bidi step** that Latin pipelines never need.
- **Real official documents, not scans.** Inputs are phone photos with **glare, skew, motion blur, low resolution, and dense decorative backgrounds** — exactly the conditions that off-the-shelf OCR fails on. Most of the engineering here is the *preprocessing* that makes the OCR call viable at all.
- **Localization is document-specific and self-anchoring.** Rather than fixed bounding boxes, the pipeline anchors on **physical document landmarks** (the CIN portrait via Haar cascade, the carte-grise red flag via HSV + dominant-color check) so it tolerates translation and scale variation across photos.
- **Layered OCR with graceful degradation.** Three engines in a fallback chain, with OCR geometry feeding back into the crop, is a deliberate response to the fact that **no single Arabic OCR engine is dependable** on this imagery.

Together this is a genuinely under-served niche: Tunisian-document Arabic field extraction with end-to-end contract generation, built from open CV/OCR components rather than a paid document-AI service.

---

## Repository layout

| File | Role |
|------|------|
| `integration_notebook.ipynb` | **End-to-end pipeline.** Orientation → Haar face anchor → field localization → K-channel threshold → layered Arabic OCR for CIN recto/verso + carte grise → JSON serialization → contract auto-fill. |
| `data_augmentation.ipynb` | **CNN document-type classifier.** Loads `carte_identite` / `carte_grise` folders, builds the augmentation stack + 3-block CNN, trains (30 epochs) and predicts the card type. |
| `conour_extraction.ipynb` | Contour detection & extraction (best on dark backgrounds): thresholding, `findContours`, bounding-box cropping. |
| `noNeeded_text_mask.ipynb` | Locates printed labels with EasyOCR and **inpaints** them out, keeping only the useful fields after a fixed resize. |
| `zoom_in.ipynb` | Computes a zoom factor from the dominant contour and rescales to enlarge the region of interest. |
| `orientation_quelque_soit_angle.ipynb` | Orientation estimation for **any** input angle via PCA (`PCACompute2`) and `minAreaRect`. |
| `projet.py` | Standalone cleanup stage: glare inpaint + 16-step adaptive brightness + label masking. |
| `crop.py` | Carte-grise crop driven by the red-flag HSV mask + K-means dominant-color anchor. |
| `Orientation.py` | CLI skew correction via Otsu + `minAreaRect` (`--image` argument). |
| `rotation.py` | Brute-force rotation search keeping the angle that maximizes OCR text length. |
| `orientaation.png` | Reference illustration for the orientation step. |

---

## Tech stack

- **Language / environment:** Python, Jupyter Notebook
- **Computer vision:** OpenCV (`opencv-python`), NumPy, SciPy (`scipy.ndimage`), `imutils`, Pillow (`PIL.ImageStat` / `ImageDraw`), Matplotlib
- **Deep learning:** TensorFlow / Keras (CNN + preprocessing-layer augmentation), scikit-learn (`train_test_split`)
- **OCR (Arabic):** EasyOCR, ArabicOCR, the [OCR.space](https://ocr.space/) HTTP API
- **Arabic text handling:** `arabic-reshaper`, `python-bidi`, `python-Levenshtein`
- **HTTP:** `requests`

---

## Getting started

```bash
# 1. Clone
git clone https://github.com/MuhamedHabib/DataScienceProject.git
cd DataScienceProject

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install the libraries used across the notebooks
pip install opencv-python numpy scipy imutils pillow matplotlib \
            tensorflow scikit-learn easyocr ArabicOcr \
            arabic-reshaper python-bidi python-Levenshtein requests

# 4. Launch Jupyter
jupyter notebook
```

Open `integration_notebook.ipynb` to run the full pipeline, or run any single notebook to explore that stage in isolation.

```bash
# Standalone scripts run directly, e.g.:
python Orientation.py --image path/to/document.jpg
```

---

## Notes, limitations & next steps

This is a working **academic / research prototype** (ESPRIT PI project). It demonstrates a complete preprocessing-and-OCR pipeline, with honest boundaries:

- **No committed evaluation set, and therefore no accuracy figures.** The repository contains **no benchmark, no labeled test split, and no reported field-level accuracy** — and this README deliberately reports none. The classifier's `model.evaluate` runs against an in-notebook split, but those numbers are not persisted or committed. **A real quantitative eval — per-field OCR accuracy (e.g. CER / exact-match per field) on a held-out, hand-labeled set of CIN and carte-grise photos — is the most important missing piece.** Without it, robustness claims are anecdotal.
- **Hard-coded local paths.** The notebooks reference absolute paths and named images (`ali1.jpg`, `verso.jpg`, `cg12.jpg`, `grise12.png`, etc.) from the author's machine. Update the paths and **bring your own document images** before running.
- **Bring your own OCR.space key.** The pipeline calls the OCR.space API; supply **your own** API key via an environment variable and **never commit it**. (Use a placeholder such as `OCR_SPACE_API_KEY` in code.)
- **No requirements lockfile.** The `pip install` above lists the libraries actually imported by the notebooks; pin versions to match your environment.
- **Heuristic thresholds.** Glare thresholds, the brightness ladder, the HSV red ranges, and the crop offsets are tuned to the sample images. A natural next step is to make localization and cleanup parameters **data-driven** (or replace the layered OCR + heuristics with a single fine-tuned Arabic document model) and to add an automated regression harness over a labeled image corpus.
- **Language note.** The notebooks were authored with French headings/comments; this README summarizes them in English. Class names like `carte_identite` (ID card) and `carte_grise` (vehicle registration) keep the original French naming.

---
<p align="center">Built by <b>Mohamed Habib Khattat</b> — <a href="https://github.com/MuhamedHabib">GitHub (@MuhamedHabib)</a> · <a href="https://www.linkedin.com/in/mohamed-habib-khattat-2b206a173">LinkedIn</a></p>
