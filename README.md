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

Add `-v` to print the full traceback when a stage fails. If no puzzle is found, the program still
writes `<name>_result.json` with `"verdict": "not_found"` and an all-zero `<name>_grid.txt`.

## Evaluation

Scores a results folder against ground-truth annotations (Appendix A format) with the Section 7
metrics, per image and per condition C1–C5:

```
python evaluate.py --results results/ --annotations SudokuScan_devset/annotations [--csv metrics.csv]
```

## Layout

| File | Stage |
|---|---|
| `sudoku_scan.py` | command-line interface, single and batch mode |
| `sudokuscan/preprocess.py` | 1 — greyscale, denoising, illumination normalisation |
| `sudokuscan/corners.py` | 2 — Harris response, candidate points, outer-corner selection |
| `sudokuscan/rectify.py` | 3 — A3, A4 and H4 rectification and residuals |
| `sudokuscan/recognize.py` | 5 — cell extraction, empty-cell test, OCR |
| `sudokuscan/validity.py` | 6 — row, column and box check |
| `sudokuscan/pipeline.py` | runs the stages on one image |
| `sudokuscan/output.py` | result files and debug images |
| `evaluate.py` | Section 7 metrics against annotations |
