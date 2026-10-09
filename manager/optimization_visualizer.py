"""
optimization_visualizer.py
The MIT License (MIT)
Copyright © 2025 Science Solutions International Laboratory, Inc.

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

# ruff: noqa: ANN001 ; typical typing cannot be used
# ruff: noqa: ARG002 ; to meet interface
import datetime
import queue
import shutil
from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Literal

import numpy as np
from dearpygui import dearpygui as dpg

from emsopt_engine.individual import Individual
from manager.gui_themes import DpgThemeRegistry, ThemeRole, build_default_theme_registry
from manager.optimization_output_handler import dump_individuals
from manager.response_surface import ResponseSurfaceRequest
from manager.restart_selection import RestartSelection

logger = getLogger(__name__)
SORT_SPEC_MIN_LEN = 2
MIN_RESPONSE_SURFACE_DESIGN_COLUMNS = 2
SO_CONSTRAINT_EPS = 1e-10


class PlotSize:
    VIEWPORT_W = 960
    VIEWPORT_H = 600
    MARGIN = 20
    WINDOW_W = VIEWPORT_W - MARGIN
    WINDOW_H = VIEWPORT_H - MARGIN
    TEXT_SPACE = 35
    _EVEN_W = WINDOW_W / 2
    PROGRESS_PLOT_W = _EVEN_W - MARGIN
    PROGRESS_PLOT_H = WINDOW_H - MARGIN - 3 * TEXT_SPACE
    IMAGE_W = (_EVEN_W - MARGIN) / 2
    _IMAGE_H = 340
    IMAGE_H = _IMAGE_H - 2 * TEXT_SPACE - MARGIN
    METRICS_TEXT_OFFSET = 10
    METRICS_PLOT_W = _EVEN_W - MARGIN
    METRICS_PLOT_H = WINDOW_H - MARGIN - 2 * TEXT_SPACE - _IMAGE_H
    RES_SUR_W = _EVEN_W - 2 * MARGIN
    RES_SUR_H = (PROGRESS_PLOT_H - 2 * TEXT_SPACE) / 2
    RES_SUR_SCALE_W = 80


class Tags:
    # Status bar related
    STATUS_TAG = "status_label"
    SUMMARY_TAG = "summary_label"
    LEFT_TABS_TAG = "left_tabs"
    GRAPH_TAB_TAG = "graph_tab"
    TABLE_TAB_TAG = "table_tab"
    # image related
    TEXTURE_TAG = "img_tex"
    WIDGET_TAG = "img_widget"
    METRICS_TEXT_TAG = "metrics_text"
    CMP_TEXTURE_TAG = "cmp_img_tex"
    CMP_WIDGET_TAG = "cmp_img_widget"
    CMP_METRICS_TEXT_TAG = "cmp_metrics_text"
    RESTART_BUTTON_TAG = "restart_button"
    # progress plot related
    PROGRESS_PLOT_TAG = "progress_plot"
    PROGRESS_X_AXIS_TAG = "progress_x_axis"
    PROGRESS_Y_AXIS_TAG = "progress_y_axis"
    BEST_LINE_TAG = "best_line"
    CONS_LINE_TAG = "cons_line"
    PARETO_TAG = "pf_scatter"
    DISPLAYED_POINT_TAG = "displayed_point_scatter"
    PINNED_POINT_TAG = "pinned_point_scatter"
    # table related
    TABLE_CONTAINER_TAG = "individuals_table_container"
    TABLE_TAG = "individuals_table"
    # response surface related
    RESPONSE_SURFACE_TAB_TAG = "response_surface_tab"
    RESPONSE_SURFACE_STATUS_TAG = "response_surface_status"
    RESPONSE_SURFACE_TARGET_COMBO_TAG = "response_surface_target_combo"
    RESPONSE_SURFACE_X_COMBO_TAG = "response_surface_x_combo"
    RESPONSE_SURFACE_Y_COMBO_TAG = "response_surface_y_combo"
    RESPONSE_SURFACE_CONTOUR_PLOT_TAG = "response_surface_contour_plot"
    RESPONSE_SURFACE_CONTOUR_X_AXIS_TAG = "response_surface_contour_x_axis"
    RESPONSE_SURFACE_CONTOUR_Y_AXIS_TAG = "response_surface_contour_y_axis"
    RESPONSE_SURFACE_COLORMAP_GROUP_TAG = "response_surface_colormap_group"
    RESPONSE_SURFACE_COLORMAP_SCALE_TAG = "response_surface_colormap_scale"
    RESPONSE_SURFACE_HEAT_TAG = "response_surface_heat"
    RESPONSE_SURFACE_SAMPLE_TAG = "response_surface_samples"
    RESPONSE_SURFACE_IMPORTANCE_PLOT_TAG = "response_surface_importance_plot"
    RESPONSE_SURFACE_IMPORTANCE_X_AXIS_TAG = "response_surface_importance_x_axis"
    RESPONSE_SURFACE_IMPORTANCE_Y_AXIS_TAG = "response_surface_importance_y_axis"
    RESPONSE_SURFACE_IMPORTANCE_BAR_TAG = "response_surface_importance_bar"
    # metrics plot related
    METRICS_X_AXIS_TAG = "metrics_x_axis"
    METRICS_Y_AXIS_TAG = "metrics_y_axis"
    METRICS_PLOT_TAG = "metrics_plot"
    METRICS_TAG = "metrics"
    CMP_METRICS_TAG = "cmp_metrics"
    # event handler
    EVENT_HANDLER_TAG = "handler"
    ANALYSIS_SERIES_PREFIX = "analysis_series"
    ANALYSIS_MEAN_TAG = "analysis_mean"
    ANALYSIS_STD_BAND_TAG = "analysis_std_band"


@dataclass
class VisualizerPayload:
    """Payload to be transfered to OptimizationVisualizer
    Args:
        type (Literal["update", "done"]): Type of the payload,
            either "update" for ongoing optimization or "done" for completion.
        elapsed (datetime.timedelta): elapsed time. Ignored when type is "done".
        eta (datetime.timedelta): estimated time remaining. Ignored when type is "done".
        individuals (list[Individual] | None): individuals to be stored. Ignored when type is "done".
    """

    type: Literal["update", "restart_ready", "response_surface_ready", "done"]
    elapsed: datetime.timedelta | None = None
    eta: datetime.timedelta | None = None
    individuals: list[Individual] | None = None
    view_mode: Literal["live", "analysis"] = "live"
    graph_table_enabled: bool = True
    graph_table_disabled_reason: str | None = None
    response_surface_manifest: dict | None = None
    response_surface_result: dict | None = None


class OptimizationVisualizer:
    """
    Visualization class for monitoring optimization process using DearPyGui.
    Supports single-objective (SO) and multi-objective (MO) optimization visualization.
    """

    def __init__(self, output_dir: str) -> None:
        """Create internal members"""
        self.output_dir = output_dir
        self.mode: Literal["SO", "MO"] | None = None
        self.num_obj: int = 1
        self.latest_individuals: list[Individual] = []
        self.displayed_individual: Individual | None = None
        self.cmp_displayed_individual: Individual | None = None
        self.current_obj_pair: list[int, int] = [0, 1]
        self.displayed_point_coords: list[int, int] | None = None
        self._image_window = None
        self._cmp_window = None
        self.outcome_idx = 1
        self._table_container = None
        self.displayed_table_indices: list[int] = []
        self._table_column_keys: list[str] = []
        self._table_column_id_to_key: dict[int | str, str] = {}
        self._table_sort_state: tuple[str, int] | None = None
        self._table_row_tags: dict[int, str] = {}
        self._restart_q = None
        self._response_surface_request_q = None
        self._restart_target_study_name: str | None = None
        self._restart_enabled = False
        self.view_mode: Literal["live", "analysis"] = "live"
        self._analysis_series_tags: list[str] = []
        self._analysis_summary_note: str | None = None
        self.so_curve_mode: Literal["generation_best", "best_so_far"] = "generation_best"
        self.graph_table_enabled = True
        self.graph_table_disabled_reason: str | None = None
        self.response_surface_manifest: dict | None = None
        self.response_surface_result: dict | None = None
        self.current_response_surface_target: str | None = None
        self.current_response_surface_x: str | None = None
        self.current_response_surface_y: str | None = None
        self._response_surface_request_serial = 0
        self._pending_response_surface_request_id: str | None = None
        self._pending_response_surface_signature: tuple | None = None
        self._response_surface_result_signature: tuple | None = None
        self._theme_registry: DpgThemeRegistry = build_default_theme_registry()

    @property
    def iterations(self) -> list[int]:
        return list(range(1, len(self.latest_individuals) + 1, 1))

    @property
    def fitnesses(self) -> list[float]:
        return [ind.metrics.fitness for ind in self.latest_individuals]

    @property
    def so_display_values(self) -> tuple[list[float], list[int]]:
        """Return the SO curve values according to the selected display mode."""
        if self.so_curve_mode == "generation_best":
            # return fitnesses and indices itself
            return self.fitnesses, list(range(len(self.fitnesses)))
        if self.view_mode == "analysis":
            # should reset indices every when each group are screened
            group_indices = self._group_indices_by_run_id().values()
            running_best_indices = [
                best_idx for indices in group_indices for best_idx in self._get_so_running_best_indices(indices)
            ]
        else:
            running_best_indices = self._get_so_running_best_indices(list(range(len(self.latest_individuals))))
        running_best = [self.latest_individuals[idx].metrics.fitness for idx in running_best_indices]
        return running_best, running_best_indices

    @property
    def cons_violations(self) -> list[float]:
        return [ind.metrics.constraint_violation for ind in self.latest_individuals]

    @property
    def so_constraint_display_values(self) -> list[float]:
        """Return SO constraint values that correspond to the selected curve mode."""
        if self.so_curve_mode == "generation_best":
            return self.cons_violations
        return [self.latest_individuals[idx].metrics.constraint_violation for idx in self.so_display_values[1]]

    @property
    def pareto_values(self) -> tuple[list[float], list[float]]:
        x = [ind.metrics.objectives[self.current_obj_pair[0]] for ind in self.latest_individuals]
        y = [ind.metrics.objectives[self.current_obj_pair[1]] for ind in self.latest_individuals]
        return [x, y]

    """run gui"""

    def run(
        self,
        mode: Literal["SO", "MO"],
        num_obj: int,
        progress_q: object,
        stop_evt,
        pause_evt,
        restart_q: object | None = None,
        response_surface_request_q: object | None = None,
        restart_target_study_name: str | None = None,
        restart_state: Literal["disabled", "enabled"] = "disabled",
        view_mode: Literal["live", "analysis"] = "live",
    ) -> None:
        """
        Main loop for running the visualization window.

        Args:
            mode: Optimization mode, either "SO" or "MO".
            num_obj: num of objectives. Not effective when mode is SO.
            progress_q: Queue for receiving VisualizerPayload updates.
            stop_evt: Event to signal stopping the visualization.
            pause_evt: Event to signal pausing the visualization.
        """

        def drain_queue() -> None:
            """
            Process payloads from the queue and update visualization accordingly.
            """
            # process payload sent via queue
            try:
                payload: VisualizerPayload = progress_q.get_nowait()
            except queue.Empty:
                return
            if payload.type == "done":
                dpg.stop_dearpygui()
            if payload.type == "update":
                self.update(payload)
            if payload.type == "restart_ready":
                self._set_restart_state("enabled")
            if payload.type == "response_surface_ready":
                self._handle_response_surface_result(payload.response_surface_result)

        if mode not in ["SO", "MO"]:
            logger.warning("No mode name %s available. Disable visualization.", mode)
            return
        if num_obj < 1:
            logger.warning("`num_obj` should be positive. Got: %d. Disable visualization.", num_obj)
            return
        self.mode = mode
        self.num_obj = num_obj
        self.view_mode = view_mode
        self._restart_q = restart_q
        self._response_surface_request_q = response_surface_request_q
        self._restart_target_study_name = restart_target_study_name
        self._restart_enabled = restart_state == "enabled"
        self.setup(stop_evt, pause_evt)
        # show and start
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            drain_queue()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    """setups"""

    def setup_progress_plot_so(self) -> None:
        """
        Set up the plot for single-objective optimization.
        Displays best fitness and constraint violation over iterations.
        """
        with dpg.group():
            with dpg.group(horizontal=True):
                dpg.add_text("Best fitness / constraint")
                dpg.add_combo(
                    label="Curve mode",
                    items=["Best So Far", "Generation Best"],
                    default_value="Generation Best",
                    width=140,
                    callback=self._on_so_curve_mode_changed,
                )
            with dpg.plot(
                label="Iteration - fitness graph",
                tag=Tags.PROGRESS_PLOT_TAG,
                width=PlotSize.PROGRESS_PLOT_W,
                height=PlotSize.PROGRESS_PLOT_H,
            ):
                dpg.add_plot_axis(dpg.mvXAxis, label="Recorded Iteration", tag=Tags.PROGRESS_X_AXIS_TAG)
                y = dpg.add_plot_axis(dpg.mvYAxis, label="Value", tag=Tags.PROGRESS_Y_AXIS_TAG)
                if self.view_mode == "live":
                    dpg.add_line_series([], [], parent=y, tag=Tags.BEST_LINE_TAG, label="best_fitness")
                    dpg.add_line_series([], [], parent=y, tag=Tags.CONS_LINE_TAG, label="constraint_violation")
                dpg.add_scatter_series([], [], parent=y, tag=Tags.DISPLAYED_POINT_TAG)  # for clicked point
                dpg.add_scatter_series([], [], parent=y, tag=Tags.PINNED_POINT_TAG)  # for pinned point
                dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)

    def setup_progress_plot_mo(self) -> None:
        """
        Set up the plot for multi-objective optimization.
        Displays Pareto front (first two objectives).
        """
        with dpg.group():
            # title and combo boxes to switch axis
            with dpg.group(horizontal=True):
                dpg.add_text("Trade-off of obj indices:")
                items = [str(i) for i in range(self.num_obj)]
                dpg.add_combo(
                    label="for x-axis,", items=items, default_value=str(0), width=40, callback=self._on_x_changed
                )
                dpg.add_combo(
                    label="for y-axis", items=items, default_value=str(1), width=40, callback=self._on_y_changed
                )
            # plot
            with dpg.plot(
                label="obj_x - obj_y trade-off graph",
                tag=Tags.PROGRESS_PLOT_TAG,
                width=PlotSize.PROGRESS_PLOT_W,
                height=PlotSize.PROGRESS_PLOT_H,
            ):
                dpg.add_plot_axis(dpg.mvXAxis, label="Objective X", tag=Tags.PROGRESS_X_AXIS_TAG)
                y = dpg.add_plot_axis(dpg.mvYAxis, label="Objective Y", tag=Tags.PROGRESS_Y_AXIS_TAG)
                if self.view_mode == "live":
                    dpg.add_scatter_series([], [], parent=y, tag=Tags.PARETO_TAG, label="pareto")
                dpg.add_scatter_series([], [], parent=y, tag=Tags.DISPLAYED_POINT_TAG)  # for clicked point
                dpg.add_scatter_series([], [], parent=y, tag=Tags.PINNED_POINT_TAG)  # for pinned point
                dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)

    def setup_table_view(self) -> None:
        """
        Set up the table container for displaying individuals.
        """
        with dpg.child_window(
            width=PlotSize.PROGRESS_PLOT_W,
            height=PlotSize.PROGRESS_PLOT_H,
            tag=Tags.TABLE_CONTAINER_TAG,
        ) as table_container:
            dpg.add_text("Individuals table will appear here.")
        self._table_container = table_container

    def setup_response_surface_view(self) -> None:
        with dpg.child_window(width=PlotSize.PROGRESS_PLOT_W, height=PlotSize.PROGRESS_PLOT_H):
            dpg.add_text("Response surface is not available.", tag=Tags.RESPONSE_SURFACE_STATUS_TAG)
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    label="Target",
                    items=[],
                    width=150,
                    tag=Tags.RESPONSE_SURFACE_TARGET_COMBO_TAG,
                    callback=self._on_response_surface_target_changed,
                )
                dpg.add_combo(
                    label="X",
                    items=[],
                    width=40,
                    tag=Tags.RESPONSE_SURFACE_X_COMBO_TAG,
                    callback=self._on_response_surface_x_changed,
                )
                dpg.add_combo(
                    label="Y",
                    items=[],
                    width=40,
                    tag=Tags.RESPONSE_SURFACE_Y_COMBO_TAG,
                    callback=self._on_response_surface_y_changed,
                )
            with dpg.group():
                with dpg.group(horizontal=True):
                    with dpg.plot(
                        label="Response surface",
                        tag=Tags.RESPONSE_SURFACE_CONTOUR_PLOT_TAG,
                        width=int(PlotSize.RES_SUR_W - PlotSize.RES_SUR_SCALE_W - PlotSize.MARGIN / 2),
                        height=PlotSize.RES_SUR_H,
                    ):
                        dpg.add_plot_axis(
                            dpg.mvXAxis,
                            label="Design variable X",
                            tag=Tags.RESPONSE_SURFACE_CONTOUR_X_AXIS_TAG,
                        )
                        dpg.add_plot_axis(
                            dpg.mvYAxis,
                            label="Design variable Y",
                            tag=Tags.RESPONSE_SURFACE_CONTOUR_Y_AXIS_TAG,
                        )
                        dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
                    with dpg.group(tag=Tags.RESPONSE_SURFACE_COLORMAP_GROUP_TAG):
                        dpg.add_colormap_scale(
                            tag=Tags.RESPONSE_SURFACE_COLORMAP_SCALE_TAG,
                            width=PlotSize.RES_SUR_SCALE_W,
                            height=PlotSize.RES_SUR_H,
                            colormap=dpg.mvPlotColormap_Viridis,
                            min_scale=0.0,
                            max_scale=1.0,
                            format="%0.6g",
                        )
                with dpg.plot(
                    label="Feature importance",
                    tag=Tags.RESPONSE_SURFACE_IMPORTANCE_PLOT_TAG,
                    width=PlotSize.RES_SUR_W,
                    height=PlotSize.RES_SUR_H,
                ):
                    dpg.add_plot_axis(
                        dpg.mvXAxis,
                        label="Design variable",
                        tag=Tags.RESPONSE_SURFACE_IMPORTANCE_X_AXIS_TAG,
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis,
                        label="Importance",
                        tag=Tags.RESPONSE_SURFACE_IMPORTANCE_Y_AXIS_TAG,
                    )

    def setup_image(self) -> None:
        """
        Set up the image display area for showing individual's outcome image.
        """
        # image space
        dpg.add_text("Individual's outcome image (click left graph to change individual)")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Save selected", callback=self._on_save_outcome_clicked)
            dpg.add_button(label="Pin to right", callback=self._on_pin_clicked)
            dpg.add_button(label="Remove pinned", callback=self._on_remove_pinned_clicked)
            dpg.add_button(
                label="Restart from selected",
                callback=self._on_restart_clicked,
                tag=Tags.RESTART_BUTTON_TAG,
                enabled=self._restart_enabled,
            )
        with dpg.group(horizontal=True):
            # create void image widget linked to texture
            with dpg.child_window(width=PlotSize.IMAGE_W, height=PlotSize.IMAGE_H) as image_window:
                dpg.add_image(Tags.TEXTURE_TAG, tag=Tags.WIDGET_TAG, show=False)
                dpg.add_text("", tag=Tags.METRICS_TEXT_TAG)
            # image space for comparison
            with dpg.child_window(width=PlotSize.IMAGE_W, height=PlotSize.IMAGE_H) as cmp_window:
                dpg.add_image(Tags.CMP_TEXTURE_TAG, tag=Tags.CMP_WIDGET_TAG, show=False)
                dpg.add_text("", tag=Tags.CMP_METRICS_TEXT_TAG)
        self._image_window = image_window
        self._cmp_window = cmp_window

    def setup_metrics_plot(self) -> None:
        """
        Setup metrics plot
        """
        dpg.add_text("Metrics of displayed outcomes (ratio with pinned as 1)")
        with dpg.plot(height=PlotSize.METRICS_PLOT_H, width=PlotSize.METRICS_PLOT_W, tag=Tags.METRICS_PLOT_TAG):
            dpg.add_plot_axis(
                dpg.mvXAxis, label="Metrics", tag=Tags.METRICS_X_AXIS_TAG, no_gridlines=True, no_tick_labels=True
            )
            y = dpg.add_plot_axis(dpg.mvYAxis, label="Value", tag=Tags.METRICS_Y_AXIS_TAG)
            dpg.add_bar_series([], [], tag=Tags.METRICS_TAG, label="Displayed (left)", parent=y)
            dpg.add_bar_series([], [], tag=Tags.CMP_METRICS_TAG, label="Pinned (right)", parent=y)
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast, outside=True)
            dpg.set_axis_limits(Tags.METRICS_X_AXIS_TAG, 0.0, 1.0)
            dpg.set_axis_limits(Tags.METRICS_Y_AXIS_TAG, 0.0, 1.0)

    def setup(self, stop_evt, pause_evt) -> None:
        """
        Set up the plot according to the optimization mode (SO or MO), as well as image.
        """

        def _stop() -> None:
            stop_evt.set()
            dpg.set_value(Tags.STATUS_TAG, "Stopped (waiting the end of current iteration)")

        def _toggle_pause() -> None:
            # disable when stop_evt is set
            if stop_evt.is_set():
                return
            # toggle pause
            if pause_evt.is_set():
                pause_evt.clear()
                dpg.set_value(Tags.STATUS_TAG, "Running")
            else:
                pause_evt.set()
                dpg.set_value(Tags.STATUS_TAG, "Paused (waiting the end of current iteration)")

        dpg.create_context()
        # register void texture for image
        with dpg.texture_registry(show=False):
            dpg.add_static_texture(1, 1, [0, 0, 0, 0], tag=Tags.TEXTURE_TAG)
            dpg.add_static_texture(1, 1, [0, 0, 0, 0], tag=Tags.CMP_TEXTURE_TAG)
        self._theme_registry.register_all()
        # register handler
        with dpg.item_handler_registry(tag=Tags.EVENT_HANDLER_TAG):
            dpg.add_item_clicked_handler(callback=self._on_plot_clicked)
        # create window and setup
        with (
            dpg.window(label="Optimization Process Monitor", width=PlotSize.WINDOW_W, height=PlotSize.WINDOW_H),
            dpg.group(),
        ):
            with dpg.group(horizontal=True):
                if self.view_mode == "live":
                    dpg.add_button(label="Stop", callback=_stop)
                    dpg.add_button(label="Pause / Resume", callback=_toggle_pause)
                    dpg.add_text("Running", tag=Tags.STATUS_TAG)
                else:
                    dpg.add_text("Analysis", tag=Tags.STATUS_TAG)
                dpg.add_text("", tag=Tags.SUMMARY_TAG)
            with dpg.group(horizontal=True):
                with dpg.tab_bar(tag=Tags.LEFT_TABS_TAG):
                    with dpg.tab(label="Graph", tag=Tags.GRAPH_TAB_TAG):
                        if self.mode == "SO":
                            self.setup_progress_plot_so()
                        elif self.mode == "MO":
                            self.setup_progress_plot_mo()
                        dpg.bind_item_handler_registry(Tags.PROGRESS_PLOT_TAG, Tags.EVENT_HANDLER_TAG)
                    with dpg.tab(label="Table", tag=Tags.TABLE_TAB_TAG):
                        self.setup_table_view()
                    with dpg.tab(label="Response Surface", tag=Tags.RESPONSE_SURFACE_TAB_TAG):
                        self.setup_response_surface_view()
                with dpg.group():
                    dpg.add_text("")
                    self.setup_image()
                    self.setup_metrics_plot()
        dpg.create_viewport(title="Optimization Process Monitor", width=PlotSize.VIEWPORT_W, height=PlotSize.VIEWPORT_H)
        dpg.setup_dearpygui()

    """updates"""

    def update_summary(self, elapsed: datetime.timedelta, eta: datetime.timedelta) -> None:
        """
        Update the summary text in the status bar.
        Args:
            elapsed (datetime.timedelta): Elapsed time
            eta (datetime.timedelta): Estimated time remaining
        """
        # trim to ignore milliseconds
        elapsed_trimmed = datetime.timedelta(seconds=int(elapsed.total_seconds()))
        eta_trimmed = datetime.timedelta(seconds=int(eta.total_seconds()))
        dpg.set_value(Tags.SUMMARY_TAG, f"Elapsed: {elapsed_trimmed} | ETA: {eta_trimmed}")

    def update_analysis_summary(self) -> None:
        """Show a short summary for static multi-run analysis mode."""
        run_ids = sorted({getattr(ind, "analysis_run_id", "") for ind in self.latest_individuals if ind})
        summary = f"Displayed runs: {len(run_ids)} | Individuals: {len(self.latest_individuals)}"
        if self._analysis_summary_note:
            summary = f"{summary} | {self._analysis_summary_note}"
        dpg.set_value(Tags.SUMMARY_TAG, summary)

    def update_plot_so(self) -> None:
        """
        Update the single-objective plot with new iteration data.
        """
        # update plot
        dpg.set_value(Tags.BEST_LINE_TAG, [self.iterations, self.so_display_values[0]])
        dpg.set_value(Tags.CONS_LINE_TAG, [self.iterations, self.so_constraint_display_values])
        dpg.fit_axis_data(Tags.PROGRESS_X_AXIS_TAG)
        dpg.fit_axis_data(Tags.PROGRESS_Y_AXIS_TAG)

    def update_plot_so_analysis(self) -> None:
        """Update single-objective convergence graph with one line per run."""
        self._clear_analysis_series()
        y_axis = Tags.PROGRESS_Y_AXIS_TAG
        grouped = self._group_indices_by_run_id()
        for run_id, indices in grouped.items():
            x_values = [getattr(self.latest_individuals[idx], "analysis_x_value", idx + 1) for idx in indices]
            y_values = self._get_so_series_values(indices)
            tag = f"{Tags.ANALYSIS_SERIES_PREFIX}_{run_id}"
            dpg.add_line_series(x_values, y_values, parent=y_axis, tag=tag, label=str(run_id))
            self._analysis_series_tags.append(tag)
        aggregate = self._build_so_analysis_aggregate(grouped)
        self._analysis_summary_note = aggregate["summary_note"]
        if aggregate["enabled"]:
            dpg.add_shade_series(
                aggregate["x_values"],
                aggregate["upper_values"],
                y2=aggregate["lower_values"],
                parent=y_axis,
                tag=Tags.ANALYSIS_STD_BAND_TAG,
                label="mean ± std",
            )
            self._analysis_series_tags.append(Tags.ANALYSIS_STD_BAND_TAG)
            dpg.add_line_series(
                aggregate["x_values"],
                aggregate["mean_values"],
                parent=y_axis,
                tag=Tags.ANALYSIS_MEAN_TAG,
                label="mean",
            )
            self._analysis_series_tags.append(Tags.ANALYSIS_MEAN_TAG)
        dpg.fit_axis_data(Tags.PROGRESS_X_AXIS_TAG)
        dpg.fit_axis_data(Tags.PROGRESS_Y_AXIS_TAG)

    def update_plot_mo(self) -> None:
        """
        Update the multi-objective plot with new Pareto front data.
        """
        # update plot
        dpg.set_value(Tags.PARETO_TAG, self.pareto_values)
        dpg.fit_axis_data(Tags.PROGRESS_X_AXIS_TAG)
        dpg.fit_axis_data(Tags.PROGRESS_Y_AXIS_TAG)

    def update_plot_mo_analysis(self) -> None:
        """Update multi-objective graph with one scatter series per run."""
        self._clear_analysis_series()
        y_axis = Tags.PROGRESS_Y_AXIS_TAG
        grouped = self._group_indices_by_run_id()
        for run_id, indices in grouped.items():
            x_values = [self.latest_individuals[idx].metrics.objectives[self.current_obj_pair[0]] for idx in indices]
            y_values = [self.latest_individuals[idx].metrics.objectives[self.current_obj_pair[1]] for idx in indices]
            tag = f"{Tags.ANALYSIS_SERIES_PREFIX}_{run_id}"
            dpg.add_scatter_series(x_values, y_values, parent=y_axis, tag=tag, label=str(run_id))
            self._analysis_series_tags.append(tag)
        dpg.fit_axis_data(Tags.PROGRESS_X_AXIS_TAG)
        dpg.fit_axis_data(Tags.PROGRESS_Y_AXIS_TAG)

    def update_table_view(self) -> None:
        """
        Rebuild the table view using the currently displayed individuals.
        """
        if self._table_container is None or not dpg.does_item_exist(self._table_container):
            return

        self._clear_table_container()

        if not self.latest_individuals:
            dpg.add_text("No individuals to display.", parent=self._table_container)
            return

        objectives_count, other_metrics_count = self._get_table_column_counts()
        selected_idx = self._get_displayed_individual_idx()

        with dpg.table(
            parent=self._table_container,
            tag=Tags.TABLE_TAG,
            header_row=True,
            row_background=True,
            sortable=True,
            resizable=True,
            borders_innerH=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_outerV=True,
            scrollY=True,
            policy=dpg.mvTable_SizingStretchProp,
            callback=self._on_table_sort,
        ):
            self._table_row_tags = {}
            column_defs = self._build_table_column_defs(objectives_count, other_metrics_count)
            self._add_table_columns(column_defs)
            for idx in self.displayed_table_indices:
                individual = self.latest_individuals[idx]
                self._add_table_row(idx, individual, selected_idx, objectives_count, other_metrics_count)

    def update_response_surface_view(self) -> None:
        """Refresh combo-box state and trigger lazy generation for the current selection."""
        if not dpg.does_item_exist(Tags.RESPONSE_SURFACE_STATUS_TAG):
            return
        if not self.response_surface_manifest:
            dpg.set_value(Tags.RESPONSE_SURFACE_STATUS_TAG, "Response surface is not available.")
            return

        targets = list(self.response_surface_manifest.get("target_columns", []))
        design_columns = list(self.response_surface_manifest.get("design_columns", []))
        if not targets or len(design_columns) < MIN_RESPONSE_SURFACE_DESIGN_COLUMNS:
            dpg.set_value(Tags.RESPONSE_SURFACE_STATUS_TAG, "Response surface is not available.")
            return

        self.current_response_surface_target = self.current_response_surface_target or targets[0]
        self.current_response_surface_x = self.current_response_surface_x or design_columns[0]
        self.current_response_surface_y = self.current_response_surface_y or design_columns[1]
        if self.current_response_surface_x == self.current_response_surface_y:
            self.current_response_surface_y = next(
                column for column in design_columns if column != self.current_response_surface_x
            )

        dpg.configure_item(Tags.RESPONSE_SURFACE_TARGET_COMBO_TAG, items=targets)
        dpg.configure_item(Tags.RESPONSE_SURFACE_X_COMBO_TAG, items=design_columns)
        dpg.configure_item(Tags.RESPONSE_SURFACE_Y_COMBO_TAG, items=design_columns)
        dpg.set_value(Tags.RESPONSE_SURFACE_TARGET_COMBO_TAG, self.current_response_surface_target)
        dpg.set_value(Tags.RESPONSE_SURFACE_X_COMBO_TAG, self.current_response_surface_x)
        dpg.set_value(Tags.RESPONSE_SURFACE_Y_COMBO_TAG, self.current_response_surface_y)
        self._request_response_surface_update()

    def _request_response_surface_update(self) -> None:
        """Send one lazy-generation request unless the same signature is already pending or cached."""
        if self.response_surface_manifest is None or self._response_surface_request_q is None:
            return
        if (
            self.current_response_surface_target is None
            or self.current_response_surface_x is None
            or self.current_response_surface_y is None
        ):
            return
        anchor_solution, anchor_source = self._get_response_surface_anchor()
        signature = (
            self.current_response_surface_target,
            self.current_response_surface_x,
            self.current_response_surface_y,
            tuple(anchor_solution) if anchor_solution is not None else None,
        )
        if self._pending_response_surface_signature == signature:
            return
        if self._response_surface_result_signature == signature and self.response_surface_result is not None:
            # Reuse the last rendered result when the GUI returns to the same target/axes/anchor.
            self._render_response_surface_result(self.response_surface_result)
            return
        self._response_surface_request_serial += 1
        request_id = str(self._response_surface_request_serial)
        request = ResponseSurfaceRequest(
            request_id=request_id,
            target_column=self.current_response_surface_target,
            x_column=self.current_response_surface_x,
            y_column=self.current_response_surface_y,
            anchor_solution=anchor_solution,
            anchor_source=anchor_source,
            grid_size=int(self.response_surface_manifest.get("grid_size", 0) or 0) or None,
        )
        try:
            self._response_surface_request_q.put_nowait(request)
        except queue.Full:
            logger.warning("Response surface request queue is full. Skip request.")
            return
        self._pending_response_surface_request_id = request_id
        self._pending_response_surface_signature = signature
        dpg.set_value(
            Tags.RESPONSE_SURFACE_STATUS_TAG,
            f"Generating... Samples: {self.response_surface_manifest.get('num_samples', 0)}",
        )

    def _get_response_surface_anchor(
        self,
    ) -> tuple[list[float] | None, Literal["selected_individual", "median_record"]]:
        """Use the selected individual as the fixed point when available, else fall back to records median."""
        if self.displayed_individual is not None and self.displayed_individual.solution:
            return list(self.displayed_individual.solution), "selected_individual"
        return None, "median_record"

    def _handle_response_surface_result(self, result: dict | None) -> None:
        """Accept only the newest manager response and ignore stale results from older requests."""
        if not isinstance(result, dict):
            return
        request_id = str(result.get("request_id", ""))
        if (
            self._pending_response_surface_request_id is not None
            and request_id != self._pending_response_surface_request_id
        ):
            return
        self._pending_response_surface_request_id = None
        if "error" in result:
            self._pending_response_surface_signature = None
            dpg.set_value(Tags.RESPONSE_SURFACE_STATUS_TAG, f"Response surface error: {result['error']}")
            return
        self.response_surface_result = result
        anchor_solution = result.get("anchor_solution")
        self._response_surface_result_signature = (
            result.get("target"),
            result.get("x"),
            result.get("y"),
            tuple(anchor_solution) if isinstance(anchor_solution, list) else None,
        )
        self._pending_response_surface_signature = None
        self._render_response_surface_result(result)

    def _render_response_surface_result(self, result: dict) -> None:
        """Translate manager-side numeric response data into Dear PyGui plot series."""
        contour = result.get("contour")
        importance = result.get("importance")
        if not isinstance(contour, dict) or not isinstance(importance, dict):
            return
        self._replace_response_surface_contour_series(contour)
        self._replace_response_surface_importance_series(importance)
        samples = self.response_surface_manifest.get("num_samples", 0) if self.response_surface_manifest else 0
        anchor_source = str(result.get("anchor_source", "median_record"))
        dpg.set_value(
            Tags.RESPONSE_SURFACE_STATUS_TAG,
            f"Samples: {samples} | Anchor: {anchor_source}",
        )

    def update_graph_table_state(self) -> None:
        if dpg.does_item_exist(Tags.GRAPH_TAB_TAG):
            dpg.configure_item(Tags.GRAPH_TAB_TAG, show=self.graph_table_enabled)
        if dpg.does_item_exist(Tags.TABLE_TAB_TAG):
            dpg.configure_item(Tags.TABLE_TAB_TAG, show=self.graph_table_enabled)
        if not self.graph_table_enabled:
            reason = self.graph_table_disabled_reason or "Graph/Table data is not available."
            dpg.set_value(Tags.SUMMARY_TAG, reason)

    def update_graph_table_views(self) -> None:
        if not self.graph_table_enabled:
            self.update_graph_table_state()
            return
        if self.mode == "SO":
            if self.view_mode == "analysis":
                self.update_plot_so_analysis()
            else:
                self.update_plot_so()
            self.change_displayed_individual(len(self.latest_individuals) - 1)
        elif self.mode == "MO":
            if self.view_mode == "analysis":
                self.update_plot_mo_analysis()
            else:
                self.update_plot_mo()
            if self.latest_individuals and self.displayed_individual not in self.latest_individuals:
                self.change_displayed_individual(0)
        self.update_table_view()

    def _replace_response_surface_contour_series(self, contour: dict) -> None:
        """Rebuild the heat-map plot for the current target/axis selection."""
        for tag in (Tags.RESPONSE_SURFACE_HEAT_TAG, Tags.RESPONSE_SURFACE_SAMPLE_TAG):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        dpg.configure_item(
            Tags.RESPONSE_SURFACE_CONTOUR_X_AXIS_TAG,
            label=str(self.current_response_surface_x or "Design variable X"),
        )
        dpg.configure_item(
            Tags.RESPONSE_SURFACE_CONTOUR_Y_AXIS_TAG,
            label=str(self.current_response_surface_y or "Design variable Y"),
        )
        dpg.add_heat_series(
            contour["z_values"],
            contour["rows"],
            contour["cols"],
            parent=Tags.RESPONSE_SURFACE_CONTOUR_Y_AXIS_TAG,
            tag=Tags.RESPONSE_SURFACE_HEAT_TAG,
            bounds_min=tuple(contour["bounds_min"]),
            bounds_max=tuple(contour["bounds_max"]),
            scale_min=float(contour["scale_min"]),
            scale_max=float(contour["scale_max"]),
            format="",
        )
        # Dear PyGui applies plot colormaps to the plot container, not to each heat series directly.
        dpg.bind_colormap(Tags.RESPONSE_SURFACE_CONTOUR_PLOT_TAG, dpg.mvPlotColormap_Viridis)
        self._replace_response_surface_colormap_scale(contour)
        sample_points = contour.get("sample_points", {})
        dpg.add_scatter_series(
            sample_points.get("x", []),
            sample_points.get("y", []),
            parent=Tags.RESPONSE_SURFACE_CONTOUR_Y_AXIS_TAG,
            tag=Tags.RESPONSE_SURFACE_SAMPLE_TAG,
            label="records",
        )
        self._theme_registry.bind(Tags.RESPONSE_SURFACE_SAMPLE_TAG, ThemeRole.RESPONSE_SURFACE_SAMPLES)
        dpg.fit_axis_data(Tags.RESPONSE_SURFACE_CONTOUR_X_AXIS_TAG)
        dpg.fit_axis_data(Tags.RESPONSE_SURFACE_CONTOUR_Y_AXIS_TAG)

    def _replace_response_surface_importance_series(self, importance: dict) -> None:
        """Rebuild the RF feature-importance bar chart for the current target."""
        if dpg.does_item_exist(Tags.RESPONSE_SURFACE_IMPORTANCE_BAR_TAG):
            dpg.delete_item(Tags.RESPONSE_SURFACE_IMPORTANCE_BAR_TAG)
        feature_names = [str(name) for name in importance.get("feature_names", [])]
        values = [float(value) for value in importance.get("values", [])]
        x_values = list(range(len(feature_names)))
        dpg.add_bar_series(
            x_values,
            values,
            parent=Tags.RESPONSE_SURFACE_IMPORTANCE_Y_AXIS_TAG,
            tag=Tags.RESPONSE_SURFACE_IMPORTANCE_BAR_TAG,
            weight=0.67,
        )
        self._theme_registry.bind(Tags.RESPONSE_SURFACE_IMPORTANCE_BAR_TAG, ThemeRole.RESPONSE_SURFACE_IMPORTANCE)
        ticks = tuple((feature_name, idx) for idx, feature_name in enumerate(feature_names))
        dpg.set_axis_ticks(Tags.RESPONSE_SURFACE_IMPORTANCE_X_AXIS_TAG, ticks)
        dpg.configure_item(
            Tags.RESPONSE_SURFACE_IMPORTANCE_Y_AXIS_TAG,
            label="Importance ratio",
        )
        dpg.fit_axis_data(Tags.RESPONSE_SURFACE_IMPORTANCE_X_AXIS_TAG)
        dpg.fit_axis_data(Tags.RESPONSE_SURFACE_IMPORTANCE_Y_AXIS_TAG)

    def _replace_response_surface_colormap_scale(self, contour: dict) -> None:
        """Recreate the colormap scale so its labels always reflect the current response values."""
        if dpg.does_item_exist(Tags.RESPONSE_SURFACE_COLORMAP_SCALE_TAG):
            dpg.delete_item(Tags.RESPONSE_SURFACE_COLORMAP_SCALE_TAG)
        dpg.add_colormap_scale(
            tag=Tags.RESPONSE_SURFACE_COLORMAP_SCALE_TAG,
            parent=Tags.RESPONSE_SURFACE_COLORMAP_GROUP_TAG,
            width=PlotSize.RES_SUR_SCALE_W,
            height=PlotSize.RES_SUR_H,
            colormap=dpg.mvPlotColormap_Viridis,
            min_scale=float(contour["scale_min"]),
            max_scale=float(contour["scale_max"]),
            format="%0.6g",
        )

    def change_displayed_individual(self, idx: int) -> None:
        """
        Change the displayed individual and linked infomation
        Args:
            idx (int): idx of individual to display
        """
        idx_point = idx
        if self.mode == "SO" and self.so_curve_mode == "best_so_far":
            idx = self.so_display_values[1][idx]
        if not self._validate_selectable_idx(idx):
            return
        self.displayed_individual = self.latest_individuals[idx]
        img_filepath = None
        if self.displayed_individual.outcome_filepath:
            img_filepath = Path(self.displayed_individual.outcome_filepath).with_suffix(".png")
        if img_filepath is not None and img_filepath.exists():
            # change image
            self.re_create_image_widget(str(img_filepath), Tags.TEXTURE_TAG, Tags.WIDGET_TAG, self._image_window)
        elif dpg.does_item_exist(Tags.WIDGET_TAG):
            dpg.configure_item(Tags.WIDGET_TAG, show=False)
        self._update_metrics_plot()
        self._update_metrics_text(
            self.displayed_individual.metrics.other_metrics,
            Tags.METRICS_TEXT_TAG,
            self._image_window,
        )
        # display clicked point stressed
        if self.mode == "SO":
            x_val = self._get_plot_x_value(idx_point)
            y_val = self.so_display_values[0][idx_point]
        elif self.mode == "MO":
            x_val = self.pareto_values[0][idx_point]
            y_val = self.pareto_values[1][idx_point]
        dpg.set_value(Tags.DISPLAYED_POINT_TAG, [[x_val], [y_val]])
        self.displayed_point_coords = (x_val, y_val)
        self._refresh_table_selection_state()
        if self.response_surface_manifest is not None:
            self._request_response_surface_update()

    def update(self, payload: VisualizerPayload) -> None:
        """
        Update plots and image for each optimization iteration.
        Args:
            payload: VisualizerPayload containing update information.
        """
        self.latest_individuals = payload.individuals
        self.view_mode = payload.view_mode
        graph_table_enabled = getattr(payload, "graph_table_enabled", True)
        self.graph_table_enabled = bool(graph_table_enabled)
        graph_table_disabled_reason = getattr(payload, "graph_table_disabled_reason", None)
        self.graph_table_disabled_reason = (
            str(graph_table_disabled_reason) if graph_table_disabled_reason is not None else None
        )
        response_surface_manifest = getattr(payload, "response_surface_manifest", None)
        self.response_surface_manifest = (
            response_surface_manifest if isinstance(response_surface_manifest, dict) else None
        )
        if self.response_surface_manifest is None:
            self.response_surface_result = None
            self._pending_response_surface_request_id = None
            self._pending_response_surface_signature = None
            self._response_surface_result_signature = None
        self._reset_table_order()
        self._apply_current_table_sort()
        try:
            if payload.elapsed is not None and payload.eta is not None:
                self.update_summary(payload.elapsed, payload.eta)
            elif self.view_mode == "analysis":
                self.update_analysis_summary()
            self.update_graph_table_views()
            if self.response_surface_manifest is not None:
                self.update_response_surface_view()
        except (ValueError, IndexError, FileNotFoundError, AttributeError) as e:
            msg = f"Error during visualization of progress. Error message: {e}"
            logger.warning(msg)

    def re_create_image_widget(self, img_filepath: str, texture_tag: str, widget_tag: str, parent_window: str) -> None:
        """
        Re-create the image widget with the specified texture and parent window.

        Args:
            img_filepath (str): Path to the image file.
            texture_tag (str): Tag for the texture to be created.
            widget_tag (str): Tag for the image widget to be created.
            parent_window (str): Parent window tag where the image widget will be placed.
        """
        # load image
        w, h, _, data = dpg.load_image(img_filepath)
        # delete current texture and image widget
        dpg.delete_item(texture_tag)
        dpg.delete_item(widget_tag)
        # re-create texture with original size
        with dpg.texture_registry(show=False):
            dpg.add_static_texture(w, h, data, tag=texture_tag)
        # re-create image widget with proper size for display keeping original ratio
        disp_w = min(PlotSize.IMAGE_W - PlotSize.MARGIN, w)
        disp_h = int(h * (disp_w / w))
        dpg.add_image(
            texture_tag,
            tag=widget_tag,
            show=True,
            width=disp_w,
            height=disp_h,
            parent=parent_window,
        )

    """interactions"""

    def _on_save_outcome_clicked(self) -> None:
        """save outcome"""
        save_dir = Path(self.output_dir) / "outcome"
        save_dir.mkdir(exist_ok=True)
        if self.displayed_individual:
            _filepath = str(save_dir / f"outcome_{self.outcome_idx}")
            if self.displayed_individual.outcome_filepath:
                # assumes png for image
                filepath_outcome = Path(self.displayed_individual.outcome_filepath)
                filepath_image = Path(self.displayed_individual.outcome_filepath).with_suffix(".png")
                if filepath_outcome.exists():
                    shutil.copy(filepath_outcome, _filepath + filepath_outcome.suffix)
                if filepath_image.exists():
                    shutil.copy(filepath_image, _filepath + filepath_image.suffix)
            dump_individuals([self.displayed_individual], _filepath + ".yaml")
            self.outcome_idx += 1
            logger.info("Outcome saved as: %s", _filepath)
        else:
            logger.info("Nothing to save.")

    def _on_plot_clicked(self) -> None:
        """Find nearest individual by clicked position, and display the individual. Also highlight the clicked point."""
        # get clicked position in plot
        px, py = dpg.get_plot_mouse_pos()
        # find nearest individual and display
        if not self.latest_individuals:
            return
        if self.mode == "SO":
            idx = min(
                range(len(self.latest_individuals)),
                key=lambda i: (self._get_plot_x_value(i) - px) ** 2 + (self._get_plot_y_value(i) - py) ** 2,
            )
        elif self.mode == "MO":
            x, y = self.pareto_values
            idx = min(range(len(x)), key=lambda i: (x[i] - px) ** 2 + (y[i] - py) ** 2)
        else:
            return
        self.change_displayed_individual(idx)

    def _on_table_row_selected(self, sender, app_data, user_data) -> None:
        """Display the individual selected from the table."""
        if not app_data:
            return
        self.change_displayed_individual(user_data)

    def _on_table_sort(self, sender, app_data, user_data) -> None:
        """Sort table rows when a header is clicked."""
        if not app_data:
            return
        sort_spec = app_data[0]
        if len(sort_spec) < SORT_SPEC_MIN_LEN:
            return
        column_id = sort_spec[0]
        direction = sort_spec[1]
        key = self._table_column_id_to_key.get(column_id)
        if key is None:
            return
        direction_sign = 1 if direction >= 0 else -1
        self._table_sort_state = (key, direction_sign)
        self._sort_table_indices(key, direction_sign)
        self._reorder_table_rows()

    def _on_pin_clicked(self) -> None:
        """pin displayed individual to comparison space"""
        if self.displayed_individual and self.displayed_individual.outcome_filepath:
            self.cmp_displayed_individual = self.displayed_individual
            # re-create image widget on compare space
            img_filepath = Path(self.cmp_displayed_individual.outcome_filepath).with_suffix(".png")
            self.re_create_image_widget(str(img_filepath), Tags.CMP_TEXTURE_TAG, Tags.CMP_WIDGET_TAG, self._cmp_window)
            self._update_metrics_plot()
            self._update_metrics_text(
                self.cmp_displayed_individual.metrics.other_metrics, Tags.CMP_METRICS_TEXT_TAG, self._cmp_window
            )
            dpg.set_value(Tags.PINNED_POINT_TAG, [[self.displayed_point_coords[0]], [self.displayed_point_coords[1]]])
        else:
            logger.info("Nothing to pin.")

    def _on_remove_pinned_clicked(self) -> None:
        """remove pinned image (just change show flag to False)"""
        if self.cmp_displayed_individual:
            dpg.configure_item(Tags.CMP_WIDGET_TAG, show=False)
            self.cmp_displayed_individual = None
            self._update_metrics_plot()
            self._update_metrics_text([], Tags.CMP_METRICS_TEXT_TAG, self._cmp_window)
            dpg.set_value(Tags.PINNED_POINT_TAG, [[], []])
        else:
            logger.info("Nothing to remove.")

    def _on_restart_clicked(self) -> None:
        """Request a restart into the configured target study."""
        self._request_restart()

    def _on_x_changed(self, sender, app_data, user_data) -> None:
        self.current_obj_pair[0] = int(app_data)
        if self.view_mode == "analysis":
            self.update_plot_mo_analysis()
        else:
            self.update_plot_mo()
        self._update_displayed_and_pinned_point()

    def _on_y_changed(self, sender, app_data, user_data) -> None:
        self.current_obj_pair[1] = int(app_data)
        if self.view_mode == "analysis":
            self.update_plot_mo_analysis()
        else:
            self.update_plot_mo()
        self._update_displayed_and_pinned_point()

    def _on_response_surface_target_changed(self, sender, app_data, user_data) -> None:
        self.current_response_surface_target = str(app_data)
        self.update_response_surface_view()

    def _on_response_surface_x_changed(self, sender, app_data, user_data) -> None:
        self.current_response_surface_x = str(app_data)
        if self.current_response_surface_y == self.current_response_surface_x:
            design_columns = list((self.response_surface_manifest or {}).get("design_columns", []))
            self.current_response_surface_y = next(
                (column for column in design_columns if column != self.current_response_surface_x),
                self.current_response_surface_y,
            )
        self.update_response_surface_view()

    def _on_response_surface_y_changed(self, sender, app_data, user_data) -> None:
        self.current_response_surface_y = str(app_data)
        if self.current_response_surface_x == self.current_response_surface_y:
            design_columns = list((self.response_surface_manifest or {}).get("design_columns", []))
            self.current_response_surface_x = next(
                (column for column in design_columns if column != self.current_response_surface_y),
                self.current_response_surface_x,
            )
        self.update_response_surface_view()

    def _update_displayed_and_pinned_point(self) -> None:
        """
        Re-draw stressed points when switching axis in MO mode
        """
        # displayed_individual
        if self.displayed_individual and self.displayed_individual in self.latest_individuals:
            idx = self.latest_individuals.index(self.displayed_individual)
            if self.mode == "SO":
                x_val = self._get_plot_x_value(idx)
                y_val = self.so_display_values[0][idx]
            elif self.mode == "MO":
                x_val = self.pareto_values[0][idx]
                y_val = self.pareto_values[1][idx]
            else:
                x_val, y_val = None, None
            if x_val is not None and y_val is not None:
                dpg.set_value(Tags.DISPLAYED_POINT_TAG, [[x_val], [y_val]])
                self.displayed_point_coords = (x_val, y_val)
        else:
            dpg.set_value(Tags.DISPLAYED_POINT_TAG, [[], []])
            self.displayed_point_coords = None

        # pinned_individual
        if self.cmp_displayed_individual and self.cmp_displayed_individual in self.latest_individuals:
            idx = self.latest_individuals.index(self.cmp_displayed_individual)
            if self.mode == "SO":
                x_val = self._get_plot_x_value(idx)
                y_val = self.so_display_values[0][idx]
            elif self.mode == "MO":
                x_val = self.pareto_values[0][idx]
                y_val = self.pareto_values[1][idx]
            else:
                x_val, y_val = None, None
            if x_val is not None and y_val is not None:
                dpg.set_value(Tags.PINNED_POINT_TAG, [[x_val], [y_val]])
        else:
            dpg.set_value(Tags.PINNED_POINT_TAG, [[], []])

    def _on_so_curve_mode_changed(self, sender, app_data, user_data) -> None:
        """Switch between per-generation best and cumulative best-so-far."""
        self.so_curve_mode = "best_so_far" if app_data == "Best So Far" else "generation_best"
        if self.view_mode == "analysis":
            self.update_plot_so_analysis()
        else:
            self.update_plot_so()
        if self.displayed_individual:
            self.change_displayed_individual(self.latest_individuals.index(self.displayed_individual))
        self._update_displayed_and_pinned_point()

    """helpers"""

    def _update_metrics_plot(self) -> None:
        if not self.displayed_individual:
            logger.warning("Nothing displayed.")
            return
        metrics = self.displayed_individual.metrics.other_metrics
        if not metrics:
            dpg.set_value(Tags.METRICS_TAG, [[], []])
            dpg.set_value(Tags.CMP_METRICS_TAG, [[], []])
            dpg.configure_item(Tags.METRICS_X_AXIS_TAG, no_tick_labels=True)
            dpg.set_axis_ticks(Tags.METRICS_X_AXIS_TAG, ())
            dpg.set_axis_limits(Tags.METRICS_X_AXIS_TAG, 0.0, 1.0)
            dpg.set_axis_limits(Tags.METRICS_Y_AXIS_TAG, 0.0, 1.0)
            return
        # update cmp metrics
        if self.cmp_displayed_individual:
            cmp_metrics = self.cmp_displayed_individual.metrics.other_metrics
            cmp_x_values = list(range(11, 10 * len(cmp_metrics) + 2, 10))
            cmp_y_values = self._normalize(cmp_metrics, cmp_metrics)
            dpg.set_value(Tags.CMP_METRICS_TAG, [cmp_x_values, cmp_y_values])
        else:
            cmp_metrics = None
            cmp_x_values = []
            cmp_y_values = []
        dpg.set_value(Tags.CMP_METRICS_TAG, [cmp_x_values, cmp_y_values])
        # update displayed metrics
        x_values = list(range(10, 10 * len(metrics) + 1, 10))
        y_values = self._normalize(metrics, cmp_metrics) if cmp_metrics else self._normalize(metrics, metrics)
        dpg.set_value(Tags.METRICS_TAG, [x_values, y_values])
        # set labels and show
        tick_labels = tuple([(f"Metrics {i + 1}", 10 * (i + 1)) for i in range(len(metrics))])
        dpg.set_axis_ticks(Tags.METRICS_X_AXIS_TAG, tick_labels)
        dpg.configure_item(Tags.METRICS_X_AXIS_TAG, no_tick_labels=False)
        # set limit with proper margin
        dpg.set_axis_limits(Tags.METRICS_X_AXIS_TAG, x_values[0] - 10, x_values[-1] + 10)
        dpg.set_axis_limits(Tags.METRICS_Y_AXIS_TAG, 0.0, max(y_values + cmp_y_values) + 0.1)

    def _update_metrics_text(self, values: list[float], text_tag: str, parent_window: str) -> None:
        """update metrics text on image window"""
        dpg.delete_item(text_tag)
        if not values:
            text = ""
        else:
            texts = [f"Metrics {i + 1}: {value:.5g}" for i, value in enumerate(values)]
            text = "\n".join(texts)
        pos = [PlotSize.METRICS_TEXT_OFFSET] * 2
        dpg.add_text(
            default_value=text,
            tag=text_tag,
            pos=pos,
            parent=parent_window,
            color=[0, 0, 255, 255],
        )

    def _normalize(self, target: list[float], normalizer: list[float]) -> list[float]:
        """Normalize target float list

        Args:
            target (list[float]): target
            normalizer (list[float]): values used to normalize

        Returns:
            list[float]: normalized target list
        """
        EPS = 1e-12
        if len(target) != len(normalizer):
            logger.warning("Length of target and normalizer doesn't match")
            return target
        target_normalized = deepcopy(target)
        for i in range(len(target)):
            if abs(normalizer[i]) > EPS:
                target_normalized[i] /= normalizer[i]
        return target_normalized

    def _validate_selectable_idx(self, idx: int) -> bool:
        """Validate whether idx points to a selectable individual."""
        if not (0 <= idx < len(self.latest_individuals)):
            return False
        return bool(self.latest_individuals[idx])

    def _get_displayed_individual_idx(self) -> int | None:
        """Return the index of the displayed individual in latest_individuals."""
        if self.displayed_individual and self.displayed_individual in self.latest_individuals:
            return self.latest_individuals.index(self.displayed_individual)
        return None

    def _reset_table_order(self) -> None:
        """Reset displayed table order to the incoming individual order."""
        self.displayed_table_indices = list(range(len(self.latest_individuals)))

    def _clear_table_container(self) -> None:
        """Remove all existing table widgets from the table container."""
        for child in dpg.get_item_children(self._table_container, 1) or []:
            dpg.delete_item(child)

    def _get_table_column_counts(self) -> tuple[int, int]:
        """Return the max objective and metric counts among displayed individuals."""
        if self.mode == "SO":
            objectives_count = 1
        else:
            objectives_count = max(len(ind.metrics.objectives) for ind in self.latest_individuals)
        other_metrics_count = max(len(ind.metrics.other_metrics) for ind in self.latest_individuals)
        return objectives_count, other_metrics_count

    def _build_table_column_defs(self, objectives_count: int, other_metrics_count: int) -> list[tuple[str, str]]:
        """Return the table columns as (label, key) pairs."""
        column_defs = [("Select", "select"), ("Index", "index")]
        if self.view_mode == "analysis":
            column_defs.append(("Run ID", "run_id"))
        if self.mode == "SO":
            column_defs.append(("Fitness", "fitness"))
        else:
            column_defs.extend((f"Obj {obj_idx + 1}", f"obj_{obj_idx}") for obj_idx in range(objectives_count))
        column_defs.extend(
            (f"Metric {metric_idx + 1}", f"metric_{metric_idx}") for metric_idx in range(other_metrics_count)
        )
        return column_defs

    def _add_table_columns(self, column_defs: list[tuple[str, str]]) -> None:
        """Add header columns to the individuals table."""
        self._table_column_keys = []
        self._table_column_id_to_key = {}
        for idx, (label, key) in enumerate(column_defs):
            column_id = dpg.add_table_column(label=label, no_sort=idx == 0)
            self._table_column_keys.append(key)
            self._table_column_id_to_key[column_id] = key

    def _add_table_row(
        self,
        idx: int,
        individual: Individual,
        selected_idx: int | None,
        objectives_count: int,
        other_metrics_count: int,
    ) -> None:
        """Add one individual row to the table."""
        row_tag = f"table_row_{idx}"
        self._table_row_tags[idx] = row_tag
        with dpg.table_row(tag=row_tag):
            dpg.add_selectable(
                label="Selected" if idx == selected_idx else "Select",
                default_value=idx == selected_idx,
                callback=self._on_table_row_selected,
                user_data=idx,
            )
            dpg.add_text(str(idx))
            if self.view_mode == "analysis":
                dpg.add_text(str(getattr(individual, "analysis_run_id", "")))
            self._add_table_objective_cells(individual, objectives_count)
            self._add_table_value_cells(individual.metrics.other_metrics, other_metrics_count)

    def _add_table_value_cells(self, values: list[float], expected_count: int) -> None:
        """Add value cells to a table row, padding blanks as needed."""
        for value in values:
            dpg.add_text(f"{value:.5g}")
        for _ in range(expected_count - len(values)):
            dpg.add_text("")

    def _add_table_objective_cells(self, individual: Individual, objectives_count: int) -> None:
        """Add objective-related cells to a table row."""
        if self.mode == "SO":
            dpg.add_text(f"{individual.metrics.fitness:.5g}")
            return
        self._add_table_value_cells(individual.metrics.objectives, objectives_count)

    def _apply_current_table_sort(self) -> None:
        """Apply the current sort state to displayed table indices."""
        if self._table_sort_state is None:
            return
        key, direction = self._table_sort_state
        self._sort_table_indices(key, direction)

    def _sort_table_indices(self, key: str, direction: int) -> None:
        """Sort displayed table indices by a selected column key."""
        reverse = direction < 0
        self.displayed_table_indices.sort(
            key=lambda idx: self._get_sort_tuple(idx, key),
            reverse=reverse,
        )

    def _get_sort_tuple(self, idx: int, key: str) -> tuple[bool, float | int]:
        """Return a sortable tuple that keeps missing values grouped after normal values."""
        value = self._get_sort_value(idx, key)
        return value is None, 0 if value is None else value

    def _get_sort_value(self, idx: int, key: str) -> float | int | str | None:
        """Return the sortable value for a table column key."""
        if not (0 <= idx < len(self.latest_individuals)):
            return None
        individual = self.latest_individuals[idx]

        value: float | int | None = None
        if key in {"select", "index"}:
            value = idx
        elif key == "run_id":
            value = str(getattr(individual, "analysis_run_id", ""))
        elif key == "fitness":
            value = individual.metrics.fitness
        elif key.startswith("obj_"):
            obj_idx = int(key.removeprefix("obj_"))
            if obj_idx < len(individual.metrics.objectives):
                value = individual.metrics.objectives[obj_idx]
        elif key.startswith("metric_"):
            metric_idx = int(key.removeprefix("metric_"))
            if metric_idx < len(individual.metrics.other_metrics):
                value = individual.metrics.other_metrics[metric_idx]
        return value

    def _reorder_table_rows(self) -> None:
        """Reorder existing table rows to match displayed_table_indices."""
        if not dpg.does_item_exist(Tags.TABLE_TAG):
            return
        row_tags = [self._table_row_tags[idx] for idx in self.displayed_table_indices if idx in self._table_row_tags]
        if row_tags:
            dpg.reorder_items(Tags.TABLE_TAG, 1, row_tags)

    def _refresh_table_selection_state(self) -> None:
        """Update select labels and selected state without rebuilding or reordering the table."""
        selected_idx = self._get_displayed_individual_idx()
        for idx, row_tag in self._table_row_tags.items():
            children = dpg.get_item_children(row_tag, 1) or []
            if not children:
                continue
            selectable_tag = children[0]
            is_selected = idx == selected_idx
            dpg.configure_item(selectable_tag, label="Selected" if is_selected else "Select")
            dpg.set_value(selectable_tag, is_selected)

    def _request_restart(self) -> None:
        """Send a restart selection to the parent process and close the GUI."""
        if self.displayed_individual is None:
            logger.info("No individual is selected for restart.")
            return
        if not self._restart_enabled:
            logger.info("Restart is only available while waiting for the GUI to close.")
            return
        if self._restart_q is None:
            logger.warning("Restart queue is not available.")
            return
        if self._restart_target_study_name is None:
            logger.warning("Restart target study is not configured.")
            return
        request = RestartSelection(
            target_study_name=self._restart_target_study_name,
            mean=list(self.displayed_individual.solution),
        )
        try:
            self._restart_q.put_nowait(request)
        except queue.Full:
            logger.warning("Restart request queue is full. Skip restart request.")
            return
        dpg.set_value(
            Tags.STATUS_TAG,
            f"Restart requested for {self._restart_target_study_name}. Please close GUI to restart.",
        )

    def _set_restart_state(self, restart_state: Literal["disabled", "enabled"]) -> None:
        """Enable or disable restart controls."""
        enabled = restart_state == "enabled"
        self._restart_enabled = enabled
        if dpg.does_item_exist(Tags.RESTART_BUTTON_TAG):
            dpg.configure_item(Tags.RESTART_BUTTON_TAG, enabled=enabled)

    def _clear_analysis_series(self) -> None:
        """Delete dynamic multi-run plot series before re-drawing them."""
        for tag in self._analysis_series_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self._analysis_series_tags = []
        self._analysis_summary_note = None

    def _group_indices_by_run_id(self) -> dict[str, list[int]]:
        """Group displayed individuals by analysis run identifier."""
        grouped: dict[str, list[int]] = {}
        for idx, individual in enumerate(self.latest_individuals):
            run_id = str(getattr(individual, "analysis_run_id", "run"))
            grouped.setdefault(run_id, []).append(idx)
        return grouped

    def _get_plot_x_value(self, idx: int) -> int | float:
        """Return the x-axis coordinate for the selected SO point."""
        if self.view_mode == "analysis":
            return getattr(self.latest_individuals[idx], "analysis_x_value", idx + 1)
        return self.iterations[idx]

    def _get_plot_y_value(self, idx: int) -> float:
        """Return the y-axis coordinate for the selected SO point."""
        return self.so_display_values[0][idx]

    def _build_so_analysis_aggregate(self, grouped: dict[str, list[int]]) -> dict[str, object]:
        """Build mean/std aggregate series for SO multi-run display.

        Aggregate display is enabled only when every run history has the same
        number of points. Otherwise, only per-run lines are shown.
        """
        if not grouped:
            return {"enabled": False, "summary_note": None}
        grouped_indices = list(grouped.values())
        lengths = {len(indices) for indices in grouped_indices}
        if len(lengths) != 1:
            return {"enabled": False, "summary_note": "Aggregate skipped: history lengths differ"}

        x_values = [
            getattr(self.latest_individuals[idx], "analysis_x_value", offset + 1)
            for offset, idx in enumerate(grouped_indices[0])
        ]
        matrix = np.asarray(
            [self._get_so_series_values(indices) for indices in grouped_indices],
            dtype=float,
        )
        mean_values = np.mean(matrix, axis=0).tolist()
        std_values = np.std(matrix, axis=0, ddof=0).tolist()
        upper_values = (np.asarray(mean_values) + np.asarray(std_values)).tolist()
        lower_values = (np.asarray(mean_values) - np.asarray(std_values)).tolist()
        return {
            "enabled": True,
            "summary_note": "Aggregate: mean ± std",
            "x_values": x_values,
            "mean_values": mean_values,
            "upper_values": upper_values,
            "lower_values": lower_values,
        }

    def _get_so_series_values(self, indices: list[int]) -> list[float]:
        """Return one SO run series in the currently selected curve mode."""
        values = [self.latest_individuals[idx].metrics.fitness for idx in indices]
        if self.so_curve_mode == "generation_best":
            return values
        return [self.latest_individuals[idx].metrics.fitness for idx in self._get_so_running_best_indices(indices)]

    def _get_so_running_best_indices(self, indices: list[int]) -> list[int]:
        """Return running SO best indices using the optimizer's constraint-aware rule."""
        running_best_indices = []
        current_best_idx = None
        for idx in indices:
            if current_best_idx is None or self._is_better_so_individual(
                self.latest_individuals[idx],
                self.latest_individuals[current_best_idx],
            ):
                current_best_idx = idx
            running_best_indices.append(current_best_idx)
        return running_best_indices

    @staticmethod
    def _is_better_so_individual(candidate: Individual, incumbent: Individual) -> bool:
        """Compare SO individuals in the same way as SOOptimizerBase."""
        candidate_feasible = candidate.metrics.constraint_violation < SO_CONSTRAINT_EPS
        incumbent_feasible = incumbent.metrics.constraint_violation < SO_CONSTRAINT_EPS
        if candidate_feasible and incumbent_feasible:
            return candidate.metrics.fitness < incumbent.metrics.fitness
        if candidate_feasible and not incumbent_feasible:
            return True
        if not candidate_feasible and incumbent_feasible:
            return False
        return candidate.metrics.constraint_violation < incumbent.metrics.constraint_violation
