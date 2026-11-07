# SudokuScan

COMP 482 Computer Vision — Project 1. Turns a phone photograph of a printed sudoku into a 9×9 digital grid:
Harris corner detection, affine and projective rectification, cell extraction and OCR.

## Setup

1. Install the Tesseract OCR engine (see `requirements.txt` for other platforms):

   ```
   sudo pacman -S tesseract tesseract-data-eng
   ```

2. Create a virtual environment and install the Python packages:

   ```
   python3 -m venv .venv
   source .venv/bin/activate        # fish: source .venv/bin/activate.fish
   pip install -r requirements.txt
   ```

Tested with Python 3.14, OpenCV 5.0, NumPy 2.5, Tesseract 5.5.

## Usage

```
python sudoku_scan.py --input photo.jpg --outdir out/
python sudoku_scan.py --batch images_dir/ --outdir results/
```

If no puzzle is found, the program still writes `<name>_result.json` with `"verdict": "not_found"`
and an all-zero `<name>_grid.txt`.

## Evaluation

Section 7 metrics against ground-truth annotations (Appendix A format), per image and per condition:

```
python evaluate.py results/ SudokuScan_devset/annotations
```

To see what Stage 1 produces: `python tools/preview_preprocess.py previews/ image1.jpg image2.jpg ...`

## Layout

| File | Contents |
|---|---|
| `sudoku_scan.py` | command line, runs the stages, writes the output files and debug images |
| `sudokuscan/preprocess.py` | Stage 1 — resize, greyscale, illumination normalisation, ink map |
| `sudokuscan/corners.py` | Stage 2 — Harris response, candidate points, outer-corner selection |
| `sudokuscan/rectify.py` | Stage 3 — A3, A4 and H4 rectification and residuals |
| `sudokuscan/recognize.py` | Stage 5 — cell extraction, empty-cell test, OCR |
| `sudokuscan/validity.py` | Stage 6 — row, column and box check |
| `evaluate.py` | Section 7 metrics |
