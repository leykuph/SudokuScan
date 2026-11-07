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
