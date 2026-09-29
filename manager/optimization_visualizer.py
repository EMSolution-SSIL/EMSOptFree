"""
optimization_visualizer.py
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

import datetime
import queue
import shutil
from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger
from multiprocessing import Event
from pathlib import Path
from typing import Literal

from dearpygui import dearpygui as dpg

from emsopt_engine.individual import Individual
from manager.optimization_output_handler import dump_individuals

logger = getLogger(__name__)


class PlotSize:
    VIEWPORT_W = 960
    VIEWPORT_H = 600
    MARGIN = 20
    WINDOW_W = VIEWPORT_W - MARGIN
    WINDOW_H = VIEWPORT_H - MARGIN
    TEXT_SPACE = 50
    _EVEN_W = WINDOW_W / 2
    PROGRESS_PLOT_W = _EVEN_W - MARGIN
    PROGRESS_PLOT_H = WINDOW_H - MARGIN - 2 * TEXT_SPACE
    IMAGE_W = (_EVEN_W - MARGIN) / 2
    _IMAGE_H = 350
    IMAGE_H = _IMAGE_H - 2 * TEXT_SPACE
    METRICS_TEXT_OFFSET = 10
    METRICS_PLOT_W = _EVEN_W - MARGIN
    METRICS_PLOT_H = WINDOW_H - MARGIN - _IMAGE_H - TEXT_SPACE


class Tags:
    # Status bar related
    STATUS_TAG = "status_label"
    SUMMARY_TAG = "summary_label"
    # image related
    TEXTURE_TAG = "img_tex"
    WIDGET_TAG = "img_widget"
    METRICS_TEXT_TAG = "metrics_text"
    CMP_TEXTURE_TAG = "cmp_img_tex"
    CMP_WIDGET_TAG = "cmp_img_widget"
    CMP_METRICS_TEXT_TAG = "cmp_metrics_text"
    # progress plot related
    PROGRESS_PLOT_TAG = "progress_plot"
    PROGRESS_X_AXIS_TAG = "progress_x_axis"
    PROGRESS_Y_AXIS_TAG = "progress_y_axis"
    BEST_LINE_TAG = "best_line"
    CONS_LINE_TAG = "cons_line"
    PARETO_TAG = "pf_scatter"
    DISPLAYED_POINT_TAG = "displayed_point_scatter"
    PINNED_POINT_TAG = "pinned_point_scatter"
    # metrics plot related
    METRICS_X_AXIS_TAG = "metrics_x_axis"
    METRICS_Y_AXIS_TAG = "metrics_y_axis"
    METRICS_PLOT_TAG = "metrics_plot"
    METRICS_TAG = "metrics"
    CMP_METRICS_TAG = "cmp_metrics"
    # event handler
    EVENT_HANDLER_TAG = "handler"


@dataclass
class VisualizerPayload:
    """Payload to be transfered to OptimizationVisualizer
    Args:
        type (Literal["update", "done"]): Type of the payload, either "update" for ongoing optimization or "done" for completion.
        elapsed (datetime.timedelta): elapsed time. Ignored when type is "done".
        eta (datetime.timedelta): estimated time remaining. Ignored when type is "done".
        individuals (list[Individual] | None): individuals to be stored. Ignored when type is "done".
    """

    type: Literal["update", "done"]
    elapsed: datetime.timedelta | None = None
    eta: datetime.timedelta | None = None
    individuals: list[Individual] | None = None


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

    @property
    def iterations(self) -> list[int]:
        return list(range(1, len(self.latest_individuals) + 1, 1))

    @property
    def fitnesses(self) -> list[float]:
        return [ind.metrics.fitness for ind in self.latest_individuals]

    @property
    def cons_violations(self) -> list[float]:
        return [ind.metrics.constraint_violation for ind in self.latest_individuals]

    @property
    def pareto_values(self) -> tuple[list[float], list[float]]:
        x = [ind.metrics.objectives[self.current_obj_pair[0]] for ind in self.latest_individuals]
        y = [ind.metrics.objectives[self.current_obj_pair[1]] for ind in self.latest_individuals]
        return [x, y]

    """run gui"""

    def run(
        self, mode: Literal["SO", "MO"], num_obj: int, progress_q: object, stop_evt: Event, pause_evt: Event
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

        if mode not in ["SO", "MO"]:
            logger.warning("No mode name %s available. Disable visualization.", mode)
            return
        if num_obj < 1:
            logger.warning("`num_obj` should be positive. Got: %d. Disable visualization.", num_obj)
            return
        self.mode = mode
        self.num_obj = num_obj
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
            dpg.add_text("Best fitness / constraint")
            with dpg.plot(
                label="Iteration - fitness graph",
                tag=Tags.PROGRESS_PLOT_TAG,
                width=PlotSize.PROGRESS_PLOT_W,
                height=PlotSize.PROGRESS_PLOT_H,
            ):
                dpg.add_plot_axis(dpg.mvXAxis, label="Recorded Iteration", tag=Tags.PROGRESS_X_AXIS_TAG)
                y = dpg.add_plot_axis(dpg.mvYAxis, label="Value", tag=Tags.PROGRESS_Y_AXIS_TAG)
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
                dpg.add_scatter_series([], [], parent=y, tag=Tags.PARETO_TAG, label="pareto")
                dpg.add_scatter_series([], [], parent=y, tag=Tags.DISPLAYED_POINT_TAG)  # for clicked point
                dpg.add_scatter_series([], [], parent=y, tag=Tags.PINNED_POINT_TAG)  # for pinned point

    def setup_image(self) -> None:
        """
        Set up the image display area for showing individual's outcome image.
        """
        # image space
        dpg.add_text("Individual's outcome image (click left graph to change individual)")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Save outcome", callback=self._on_save_outcome_clicked)
            dpg.add_button(label="Pin to right", callback=self._on_pin_clicked)
            dpg.add_button(label="Remove pinned", callback=self._on_remove_pinned_clicked)
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

    def setup(self, stop_evt: Event, pause_evt: Event) -> None:
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
        # register handler
        with dpg.item_handler_registry(tag=Tags.EVENT_HANDLER_TAG):
            dpg.add_item_clicked_handler(callback=self._on_plot_clicked)
        # create window and setup
        with (
            dpg.window(label="Optimization Process Monitor", width=PlotSize.WINDOW_W, height=PlotSize.WINDOW_H),
            dpg.group(),
        ):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Stop", callback=_stop)
                dpg.add_button(label="Pause / Resume", callback=_toggle_pause)
                dpg.add_text("Running", tag=Tags.STATUS_TAG)
                dpg.add_text("", tag=Tags.SUMMARY_TAG)
            with dpg.group(horizontal=True):
                if self.mode == "SO":
                    self.setup_progress_plot_so()
                elif self.mode == "MO":
                    self.setup_progress_plot_mo()
                dpg.bind_item_handler_registry(Tags.PROGRESS_PLOT_TAG, Tags.EVENT_HANDLER_TAG)
                with dpg.group():
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

    def update_plot_so(self) -> None:
        """
        Update the single-objective plot with new iteration data.
        """
        # update plot
        dpg.set_value(Tags.BEST_LINE_TAG, [self.iterations, self.fitnesses])
        dpg.set_value(Tags.CONS_LINE_TAG, [self.iterations, self.cons_violations])
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

    def change_displayed_individual(self, idx: int) -> None:
        """
        Change the displayed individual and linked infomation
        Args:
            idx (int): idx of individual to display
        """
        if not self._validate_idx(idx):
            return
        img_filepath = Path(self.latest_individuals[idx].outcome_filepath).with_suffix(".png")
        if img_filepath.exists():
            self.displayed_individual = self.latest_individuals[idx]
            # change image
            self.re_create_image_widget(str(img_filepath), Tags.TEXTURE_TAG, Tags.WIDGET_TAG, self._image_window)
            self._update_metrics_plot()
            self._update_metrics_text(
                self.displayed_individual.metrics.other_metrics, Tags.METRICS_TEXT_TAG, self._image_window
            )
        else:
            logger.warning("Image file expected, but not found.")
        # display clicked point stressed
        if self.mode == "SO":
            x_val = self.iterations[idx]
            y_val = self.fitnesses[idx]
        elif self.mode == "MO":
            x_val = self.pareto_values[0][idx]
            y_val = self.pareto_values[1][idx]
        dpg.set_value(Tags.DISPLAYED_POINT_TAG, [[x_val], [y_val]])
        self.displayed_point_coords = (x_val, y_val)

    def update(self, payload: VisualizerPayload) -> None:
        """
        Update plots and image for each optimization iteration.
        Args:
            payload: VisualizerPayload containing update information.
        """
        self.latest_individuals = payload.individuals
        try:
            if payload.elapsed is not None and payload.eta is not None:
                self.update_summary(payload.elapsed, payload.eta)
            if self.mode == "SO":
                self.update_plot_so()
                # show latest individual
                self.change_displayed_individual(len(self.latest_individuals) - 1)
            elif self.mode == "MO":
                self.update_plot_mo()
        except Exception as e:
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
        disp_w = min(PlotSize.IMAGE_W, w)
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
            # assumes png for image
            filepath_outcome = Path(self.displayed_individual.outcome_filepath)
            filepath_image = Path(self.displayed_individual.outcome_filepath).with_suffix(".png")
            _filepath = str(save_dir / f"outcome_{self.outcome_idx}")
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
            idx = min(range(len(self.iterations)), key=lambda i: abs(self.iterations[i] - px))
        elif self.mode == "MO":
            x, y = self.pareto_values
            idx = min(range(len(x)), key=lambda i: (x[i] - px) ** 2 + (y[i] - py) ** 2)
        else:
            return
        self.change_displayed_individual(idx)

    def _on_pin_clicked(self) -> None:
        """pin displayed individual to comparison space"""
        if self.displayed_individual:
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

    def _on_x_changed(self, sender, app_data, user_data) -> None:
        self.current_obj_pair[0] = int(app_data)
        self.update_plot_mo()
        self._update_displayed_and_pinned_point()

    def _on_y_changed(self, sender, app_data, user_data) -> None:
        self.current_obj_pair[1] = int(app_data)
        self.update_plot_mo()
        self._update_displayed_and_pinned_point()

    def _update_displayed_and_pinned_point(self) -> None:
        """
        Re-draw stressed points when switching axis in MO mode
        """
        # displayed_individual
        if self.displayed_individual and self.displayed_individual in self.latest_individuals:
            idx = self.latest_individuals.index(self.displayed_individual)
            if self.mode == "SO":
                x_val = self.iterations[idx]
                y_val = self.fitnesses[idx]
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
                x_val = self.iterations[idx]
                y_val = self.fitnesses[idx]
            elif self.mode == "MO":
                x_val = self.pareto_values[0][idx]
                y_val = self.pareto_values[1][idx]
            else:
                x_val, y_val = None, None
            if x_val is not None and y_val is not None:
                dpg.set_value(Tags.PINNED_POINT_TAG, [[x_val], [y_val]])
        else:
            dpg.set_value(Tags.PINNED_POINT_TAG, [[], []])

    """helpers"""

    def _update_metrics_plot(self) -> None:
        if not self.displayed_individual:
            logger.warning("Nothing displayed.")
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
        metrics = self.displayed_individual.metrics.other_metrics
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
        if len(target) != len(normalizer):
            logger.warning("Length of target and normalizer doesn't match")
            return target
        target_normalized = deepcopy(target)
        for i in range(len(target)):
            target_normalized[i] /= normalizer[i]
        return target_normalized

    def _validate_idx(self, idx: int) -> bool:
        """validate if idx is valid for latest_individuals"""
        if not (0 <= idx < len(self.latest_individuals)):
            return False
        if not self.latest_individuals[idx]:
            return False
        if not self.latest_individuals[idx].outcome_filepath:  # noqa: SIM103
            return False
        return True
