"""Side detector input contracts shared by validation and restoration."""
from pathlib import Path

import numpy as np


def load_side_signal(path: Path, expected_size: int) -> np.ndarray:
    """Read one value per line; only detector count is checked semantically."""
    rows = path.read_text().splitlines()
    try:
        values = np.array([float(row) for row in rows], dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"{path}: ожидался один столбец чисел без заголовка") from error
    if len(values) != expected_size:
        raise ValueError(f"{path}: число строк {len(values)} не равно числу углов {expected_size}")
    return values
