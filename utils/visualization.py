"""
visualization.py
The MIT License (MIT)
Copyright © 2025 Sicence Solutions International Laboratory, Inc.

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

import matplotlib as mpl

# for stable image output
mpl.use("Agg")

import matplotlib.pyplot as plt


def draw_3d_surface(
    x: list[list[float]],
    y: list[list[float]],
    z: list[list[float]],
    filepath: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Function to plot a surface in 3D coordinates.

    Args:
        X: 2D array of x coordinates
        Y: 2D array of y coordinates
        Z: 2D array of z values (surface height)
        filepath (str, optional): filepath to save plot
        xlim, ylim (tuple[float, float], optional): Axis limits for x, y axes. Each should be (min, max) or None.
    Returns:
        None
    """
    # Create plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Draw 3D surface
    surf = ax.plot_surface(x, y, z, cmap="viridis", edgecolor="none")

    # Set axis labels and title
    ax.set_title("3D Surface Plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Add color bar
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.savefig(filepath)
    plt.close()


def draw_2d_scatter(
    x: list[float],
    y: list[float],
    filepath: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Function to plot a scatter plot in 2D coordinates.

    Args:
        X: 1D array of x coordinates
        Y: 1D array of y coordinates
        filepath (str, optional): filepath to save plot
        xlim, ylim (tuple, optional): Axis limits for x and y axes. Each should be (min, max) or None.
    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)
    plt.grid(visible=True)

    # Set axis limits if provided
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.savefig(filepath)
    plt.close()
