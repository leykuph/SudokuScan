"""Stage 6 - check the recognised givens against the sudoku rules."""


def groups():
    """The 27 groups of 9 cells that must not repeat a digit: 9 rows, 9 columns, 9 boxes."""
    rows = [[(r, c) for c in range(9)] for r in range(9)]
    cols = [[(r, c) for r in range(9)] for c in range(9)]
    boxes = [[(br + r, bc + c) for r in range(3) for c in range(3)]
             for br in (0, 3, 6) for bc in (0, 3, 6)]
    return rows + cols + boxes


def find_conflicts(grid):
    """Return a sorted list of (row, col) for every cell whose digit repeats in its row, column or box."""
    conflicts = set()
    for group in groups():
        for digit in range(1, 10):               # 0 = empty, never a conflict
            cells = [(r, c) for r, c in group if grid[r][c] == digit]
            if len(cells) > 1:
                conflicts.update(cells)
    return sorted(conflicts)
