"""Stage 6 - check the recognised givens against the sudoku rules."""

import numpy as np


def find_conflicts(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return the (row, col) of every cell whose digit repeats in its row, column or 3x3 box."""
    raise NotImplementedError("Stage 6 (validity check) is not implemented yet")
