"""SudokuScan command-line interface.

    python sudoku_scan.py --input photo.jpg --outdir out/
    python sudoku_scan.py --batch images_dir/ --outdir results/
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

from sudokuscan.output import write_outputs
from sudokuscan.pipeline import ScanResult, scan

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def process_file(path: Path, outdir: Path, verbose: bool = False) -> ScanResult:
    t0 = time.perf_counter()
    # imread applies the EXIF orientation tag, so phone photos come out the way they were taken.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        res = ScanResult(error=f"could not read {path}")
    else:
        res = scan(image, verbose=verbose)
    res.runtime_s = time.perf_counter() - t0
    write_outputs(outdir, path.name, res)
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read a photographed sudoku into a 9x9 grid.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="a single photograph")
    source.add_argument("--batch", type=Path, help="a folder of photographs")
    parser.add_argument("--outdir", type=Path, required=True, help="folder for the output files")
    parser.add_argument("-v", "--verbose", action="store_true", help="print tracebacks of failing stages")
    args = parser.parse_args(argv)

    paths = [args.input] if args.input else list_images(args.batch)
    args.outdir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        try:
            res = process_file(path, args.outdir, args.verbose)
        except Exception as e:  # last line of defence: one bad image must not stop the batch
            print(f"{path.name}: failed to write outputs: {e}", file=sys.stderr)
            continue
        line = f"{path.name}: {res.verdict} ({res.runtime_s:.2f} s)"
        if res.error:
            line += f"  [{res.error}]"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
