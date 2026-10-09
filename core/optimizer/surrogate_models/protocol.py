"""
protocol.py
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

from typing import Protocol

import numpy as np


class SurrogateProtocol(Protocol):
    """Minimal contract for a surrogate model"""

    # return if the surrgate model is ready to be used
    def is_ready(self) -> bool: ...
    # fit surrogate model to train data (x, y)
    def fit(self, x: np.ndarray, y: np.ndarray, **kw: object) -> None: ...
    # update (or, partially fit) surrogate model with train data (x, y)
    def update(self, x: np.ndarray, y: np.ndarray) -> None: ...
    # predict output y from input x using trained surrogate model
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]: ...  # retrun value: (mu, sigma)
