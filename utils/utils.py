"""
utils.py
The MIT License (MIT)
Copyright © 2026 Science Solutions International Laboratory, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import csv
from pathlib import Path

import numpy as np


def read_variable_length_int_csv(filepath: str | Path) -> list[np.ndarray]:
    path = Path(filepath)
    try:
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))
    except FileNotFoundError as e:
        msg = f"Variable-length CSV file not found: {path}"
        raise FileNotFoundError(msg) from e
    except OSError as e:
        msg = f"Failed to read variable-length CSV file: {path}"
        raise OSError(msg) from e

    arrays: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        try:
            values = [int(value) for value in row if value != ""]
        except ValueError as e:
            msg = f"Invalid integer in variable-length CSV file: path={path}, row={row_index}"
            raise ValueError(msg) from e
        arrays.append(np.asarray(values, dtype=np.int32))
    return arrays


def read_variable_length_float_csv(filepath: str | Path) -> list[np.ndarray]:
    path = Path(filepath)
    try:
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))
    except FileNotFoundError as e:
        msg = f"Variable-length CSV file not found: {path}"
        raise FileNotFoundError(msg) from e
    except OSError as e:
        msg = f"Failed to read variable-length CSV file: {path}"
        raise OSError(msg) from e

    arrays: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        try:
            values = [float(value) for value in row if value != ""]
        except ValueError as e:
            msg = f"Invalid float in variable-length CSV file: path={path}, row={row_index}"
            raise ValueError(msg) from e
        arrays.append(np.asarray(values, dtype=np.float64))
    return arrays
