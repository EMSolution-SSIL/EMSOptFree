"""
gui_themes.py
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

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from dearpygui import dearpygui as dpg


@dataclass(frozen=True)
class ThemeColorSpec:
    """A single theme color entry."""

    target: int
    value: tuple[int, ...]
    category: int = dpg.mvThemeCat_Plots


@dataclass(frozen=True)
class ThemeStyleSpec:
    """A single theme style entry."""

    target: int
    value1: int | float
    value2: int | float | None = None
    category: int = dpg.mvThemeCat_Plots


@dataclass(frozen=True)
class ThemeDefinition:
    """Full Dear PyGui theme definition for one semantic role."""

    component_type: int
    colors: tuple[ThemeColorSpec, ...] = ()
    styles: tuple[ThemeStyleSpec, ...] = ()


class ThemeRole(StrEnum):
    """Semantic names for reusable visual themes."""

    RESPONSE_SURFACE_IMPORTANCE = "response_surface_importance"
    RESPONSE_SURFACE_SAMPLES = "response_surface_samples"


DEFAULT_THEME_TAG_PREFIX: Final[str] = "emsopt_theme"
DEFAULT_THEME_DEFINITIONS: Final[dict[ThemeRole, ThemeDefinition]] = {
    ThemeRole.RESPONSE_SURFACE_IMPORTANCE: ThemeDefinition(
        component_type=dpg.mvBarSeries,
        colors=(
            ThemeColorSpec(dpg.mvPlotCol_Fill, (82, 145, 204)),
            ThemeColorSpec(dpg.mvPlotCol_Line, (82, 145, 204)),
        ),
    ),
    ThemeRole.RESPONSE_SURFACE_SAMPLES: ThemeDefinition(
        component_type=dpg.mvScatterSeries,
        colors=(
            ThemeColorSpec(dpg.mvPlotCol_MarkerFill, (30, 30, 30, 160)),
            ThemeColorSpec(dpg.mvPlotCol_MarkerOutline, (30, 30, 30, 200)),
        ),
        styles=(
            ThemeStyleSpec(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle),
            ThemeStyleSpec(dpg.mvPlotStyleVar_MarkerSize, 3.0),
            ThemeStyleSpec(dpg.mvPlotStyleVar_MarkerWeight, 0.5),
        ),
    ),
}


class DpgThemeRegistry:
    """Register and bind named Dear PyGui themes from semantic definitions."""

    def __init__(
        self,
        definitions: dict[ThemeRole, ThemeDefinition],
        tag_prefix: str = DEFAULT_THEME_TAG_PREFIX,
    ) -> None:
        self._definitions = definitions
        self._tag_prefix = tag_prefix

    def tag_for(self, role: ThemeRole) -> str:
        """Return the deterministic item tag used for a theme role."""
        return f"{self._tag_prefix}_{role.value}"

    def register_all(self) -> None:
        """Register every configured theme after Dear PyGui context creation."""
        for role in self._definitions:
            self.register(role)

    def register(self, role: ThemeRole) -> str:
        """Register one theme if it has not been created yet."""
        tag = self.tag_for(role)
        if dpg.does_item_exist(tag):
            return tag

        definition = self._definitions[role]
        with dpg.theme(tag=tag), dpg.theme_component(definition.component_type):
            for color_spec in definition.colors:
                dpg.add_theme_color(
                    color_spec.target,
                    color_spec.value,
                    category=color_spec.category,
                )
            for style_spec in definition.styles:
                if style_spec.value2 is None:
                    dpg.add_theme_style(
                        style_spec.target,
                        style_spec.value1,
                        category=style_spec.category,
                    )
                else:
                    dpg.add_theme_style(
                        style_spec.target,
                        style_spec.value1,
                        style_spec.value2,
                        category=style_spec.category,
                    )
        return tag

    def bind(self, item_tag: str | int, role: ThemeRole) -> None:
        """Bind a registered theme to a Dear PyGui item."""
        dpg.bind_item_theme(item_tag, self.register(role))


def build_default_theme_registry() -> DpgThemeRegistry:
    """Create the shared theme registry used by optimization visualizers."""
    return DpgThemeRegistry(DEFAULT_THEME_DEFINITIONS)
