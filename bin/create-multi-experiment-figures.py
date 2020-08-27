#! /usr/bin/env python

"""Create multi-experiment figures for the Rainbow results."""

import dataclasses
import functools
import logging
import operator
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import click
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import tqdm

from rainbow import settings, utils


logger = logging.getLogger(__name__)


# constants

N_TO_ROWS_AND_COLS = {
    1: (1, 1),
    2: (1, 2),
    3: (1, 3),
    4: (2, 2),
    5: (2, 3),
    6: (2, 3),
    7: (4, 2),
    8: (4, 2),
    9: (3, 3),
    10: (4, 3),
    11: (4, 3),
    12: (4, 3),
}
"""A mapping from various integers to a number of rows and columns.

This mapping is helpful for creating figures with multiple subfigures
when there's no obvious number of rows and columns.
"""


CENTERLINE_STYLE_KWARGS = {"c": "0.10", "linestyle": ":"}
"""Key-word arguments for styling the y = x lines in the plots."""


LINESTYLES = ["-", "--", "-.", ":"]
"""Line styles to use in the figures."""


# helper functions


def _make_plot_grid(
    plot_func: Callable,
    x_label: str,
    y_label: str,
    control_data: pd.DataFrame,
    treatment_data: pd.DataFrame,
    score_col: str,
    match_col: str,
    match_fmt: Optional[Callable],
    group_col: Optional[str],
    group_fmt: Optional[Callable],
    group_order: Optional[Callable],
    subfigure_col: Optional[str],
    subfigure_fmt: Optional[Callable],
    subfigure_order: Optional[Callable],
) -> Tuple[plt.Figure, plt.Axes]:
    """Return the grid of plots.

    Parameters
    ----------
    plot_func : Callable, required
        A function taking the ``group_to_data`` dictionary and an ``ax``
        axis object.
    x_label : str, required
        The label for the x-axis (control).
    y_label : str, required
        The label for the y-axis (treatment).
    control_data : pd.DataFrame, required
        The data for the control.
    treatment_data : pd.DataFrame, required
        The data for the treatments.
    score_col : str, required
        The name of the column containing the score.
    match_col : str, required
        The column to use for matching control and treatment scores
        together.
    match_fmt : Optional[Callable], required
        A function to apply to the match column.
    group_col : Optional[str], required
        The column to use as a key for coloring the points, which will
        be labeled in the legend (e.g. the treatments).
    group_fmt : Optional[Callable], required
        A function to format group labels for the legend.
    group_order : Optional[Callable], required
        A function to be called as a sort key when ordering the groups.
    subfigure_col : Optional[str], required
        The column to use to split the figure up into subfigures.
    subfigure_fmt : Optional[Callable], required
        A function to format the title for each subfigure.
    subfigure_order : Optional[Callable], required
        A function to be called as a sort key when ordering the
        subfigures.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray[plt.Axes]]
        A tuple containing the figure and its axes.
    """
    # Replace null function arguments with the identity.
    def identity(x):
        return x

    match_fmt = match_fmt or identity
    group_fmt = group_fmt or identity
    group_order = group_order or identity
    subfigure_fmt = subfigure_fmt or identity
    subfigure_order = subfigure_order or identity

    # Construct important constants.
    groups = (
        sorted(treatment_data[group_col].unique(), key=group_order)
        if group_col is not None
        else [None]
    )
    subfigures = (
        sorted(treatment_data[subfigure_col].unique(), key=subfigure_order)
        if subfigure_col is not None
        else [None]
    )
    n_rows, n_cols = N_TO_ROWS_AND_COLS[len(subfigures)]

    # Initialize the figure.
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(6 * n_cols, 6 * n_rows),
        constrained_layout=True,
    )
    # Modify axes so we can access the axis objects in a uniform way,
    # regardless of the number of rows or columns.
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(n_rows, n_cols)

    # Plot the subfigures.
    for i, subfigure in enumerate(subfigures):
        control_subdata = (
            control_data[control_data[subfigure_col] == subfigure]
            if subfigure_col is not None
            else control_data
        )
        treatment_subdata = (
            treatment_data[treatment_data[subfigure_col] == subfigure]
            if subfigure_col is not None
            else treatment_data
        )
        # Compute the data to plot for each group.
        group_to_data = {}
        for group in groups:
            # Join the treatment and control data for the group.
            pairs = pd.merge(
                treatment_subdata[treatment_subdata[group_col] == group]
                if group_col is not None
                else treatment_subdata,
                control_subdata,
                how="left",
                on=match_col,
                suffixes=("_treatment", "_control"),
            )
            group_to_data[group_fmt(group)] = {
                "matches": pairs[match_col].apply(match_fmt).values,
                "control_scores": pairs[f"{score_col}_control"].values,
                "treatment_scores": pairs[f"{score_col}_treatment"].values,
            }
        plot_func(
            group_to_data=group_to_data, ax=axes[i // n_cols][i % n_cols],
        )

        # Display the legend.
        axes[i // n_cols][i % n_cols].legend()
        # Set the subfigure's title.
        axes[i // n_cols][i % n_cols].set_title(subfigure_fmt(subfigure))

    for i in range(n_cols):
        axes[-1][i].set_xlabel(x_label)
    for i in range(n_rows):
        axes[i][0].set_ylabel(y_label)

    return fig, axes


def plot_paired_performance(
    control_data: pd.DataFrame,
    treatment_data: pd.DataFrame,
    score_col: str,
    match_col: str,
    match_fmt: Optional[Callable],
    group_col: Optional[str],
    group_fmt: Optional[Callable],
    group_order: Optional[Callable],
    subfigure_col: Optional[str],
    subfigure_fmt: Optional[Callable],
    subfigure_order: Optional[Callable],
) -> Tuple[plt.Figure, plt.Axes]:
    """Return the paired performance plot.

    Parameters
    ----------
    control_data : pd.DataFrame, required
        The data for the controls.
    treatment_data : pd.DataFrame, required
        The data for the treatments.
    score_col : str, required
        The name of the column containing the score.
    match_col : str, required
        The column to use for matching control and treatment scores
        together (e.g., the task).
    match_fmt : Optional[Callable], required
        A function to apply to the match column.
    group_col : Optional[str], required
        The column to use as a key for coloring the points, which will
        be labeled in the legend (e.g. the treatments).
    group_fmt : Optional[Callable], required
        A function to format group labels for the legend.
    group_order : Optional[Callable], required
        A function to be called as a sort key when ordering the groups.
    subfigure_col : Optional[str], required
        The column to use to split the figure up into subfigures.
    subfigure_fmt : Optional[str], required
        A function to format the title for each subfigure.
    subfigure_order : Optional[Callable], required
        A function to be called as a sort key when ordering the
        subfigures.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray[plt.Axes]]
        A tuple containing the figure and its axes.
    """

    def plot_func(group_to_data, ax):
        for group, data in group_to_data.items():
            matches = data["matches"]
            control_scores = data["control_scores"]
            treatment_scores = data["treatment_scores"]

            ax.scatter(control_scores, treatment_scores, label=group)
            for match, x, y in zip(matches, control_scores, treatment_scores):
                ax.annotate(match, (x, y))

        # Scale the x and y limits
        min_score = min(
            min(data["treatment_scores"].min(), data["control_scores"].min())
            for data in group_to_data.values()
        )
        max_score = max(
            max(data["treatment_scores"].max(), data["control_scores"].max())
            for data in group_to_data.values()
        )
        ax.set_xlim(0.99 * min_score, 1.01 * max_score)
        ax.set_ylim(0.99 * min_score, 1.01 * max_score)
        # Plot the y = x line.
        ax.plot(
            [0.99 * min_score, 1.01 * max_score],
            [0.99 * min_score, 1.01 * max_score],
            **CENTERLINE_STYLE_KWARGS,
        )

    return _make_plot_grid(
        plot_func=plot_func,
        x_label="control score (accuracy)",
        y_label="treatment score (accuracy)",
        control_data=control_data,
        treatment_data=treatment_data,
        score_col=score_col,
        match_col=match_col,
        match_fmt=match_fmt,
        group_col=group_col,
        group_fmt=group_fmt,
        group_order=group_order,
        subfigure_col=subfigure_col,
        subfigure_fmt=subfigure_fmt,
        subfigure_order=subfigure_order,
    )


def plot_cost_equivalent_curves(
    control_data: pd.DataFrame,
    treatment_data: pd.DataFrame,
    score_col: str,
    match_col: str,
    match_fmt: Optional[Callable],
    group_col: Optional[str],
    group_fmt: Optional[Callable],
    group_order: Optional[Callable],
    subfigure_col: Optional[str],
    subfigure_fmt: Optional[Callable],
    subfigure_order: Optional[Callable],
) -> Tuple[plt.Figure, plt.Axes]:
    """Return the cost equivalent curve plot.

    Parameters
    ----------
    control_data : pd.DataFrame, required
        The data for the control.
    treatment_data : pd.DataFrame, required
        The data for the treatments.
    score_col : str, required
        The name of the column containing the score.
    match_col : str, required
        The column to use for matching control and treatment scores
        together (e.g., the training data size).
    match_fmt : Optional[Callable], required
        Unused.
    group_col : Optional[str], required
        The column to use as a key for coloring the points, which will
        be labeled in the legend (e.g. the treatments).
    group_fmt : Optional[Callable], required
        A function to format group labels for the legend.
    group_order : Optional[Callable], required
        A function to be called as a sort key when ordering the groups.
    subfigure_col : Optional[str], required
        The column to use to split the figure up into subfigures.
    subfigure_fmt : Optional[str], required
        A function to format the title for each subfigure.
    subfigure_order : Optional[Callable], required
        A function to be called as a sort key when ordering the
        subfigures.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray[plt.Axes]]
        A tuple containing the figure and its axes.
    """

    def plot_func(group_to_data, ax):
        for i, (group, data) in enumerate(group_to_data.items()):
            matches = data["matches"]
            control_scores = data["control_scores"]
            treatment_scores = data["treatment_scores"]

            # Fit isotonic curves for the control and treatment learning
            # curves, as well as the isotonic curves' inverses.
            min_size = matches.min()
            max_size = matches.max()

            xs = np.linspace(min_size, max_size, num=2500)

            control_smoother = IsotonicRegression(out_of_bounds="clip").fit(
                matches, control_scores
            )

            treatment_smoother = IsotonicRegression(out_of_bounds="clip").fit(
                matches, treatment_scores
            )
            treatment_smoother_inv = IsotonicRegression(
                out_of_bounds="clip"
            ).fit(treatment_smoother.predict(xs), xs)

            # Plot the cost equivalent curve.
            ax.plot(
                xs,
                treatment_smoother_inv.predict(control_smoother.predict(xs)),
                linestyle=LINESTYLES[i],
                label=group,
            )
            # Plot the original data points after smoothing, since we
            # can't align them without smoothing.
            ax.scatter(
                matches,
                treatment_smoother_inv.predict(
                    control_smoother.predict(matches)
                ),
                c="k",
                s=8,
            )

        # Scale the x and y limits
        ax.set_xlim(0.99 * min_size, 1.01 * max_size)
        ax.set_ylim(0.99 * min_size, 1.01 * max_size)
        # Plot the y = x line.
        ax.plot(
            [0.99 * min_size, 1.01 * max_size],
            [0.99 * min_size, 1.01 * max_size],
            **CENTERLINE_STYLE_KWARGS,
        )
        # Add the second axis at the top of the figure.
        def cost2perf(x):
            if len(x) == 0:
                return x
            return control_smoother.predict(x.reshape(-1)).tolist()

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticklabels([f"{x:.3f}" for x in cost2perf(ax2.get_xticks())])

    return _make_plot_grid(
        plot_func=plot_func,
        x_label="control cost (# examples)",
        y_label="treatment cost (# examples)",
        control_data=control_data,
        treatment_data=treatment_data,
        score_col=score_col,
        match_col=match_col,
        match_fmt=None,
        group_col=group_col,
        group_fmt=group_fmt,
        group_order=group_order,
        subfigure_col=subfigure_col,
        subfigure_fmt=subfigure_fmt,
        subfigure_order=subfigure_order,
    )


def plot_performance_equivalent_curves(
    control_data: pd.DataFrame,
    treatment_data: pd.DataFrame,
    score_col: str,
    match_col: str,
    match_fmt: Optional[Callable],
    group_col: Optional[str],
    group_fmt: Optional[Callable],
    group_order: Optional[Callable],
    subfigure_col: Optional[str],
    subfigure_fmt: Optional[Callable],
    subfigure_order: Optional[Callable],
) -> Tuple[plt.Figure, plt.Axes]:
    """Return the performance equivalent curve plot.

    Parameters
    ----------
    control_data : pd.DataFrame, required
        The data for the controls.
    treatment_data : pd.DataFrame, required
        The data for the treatments.
    score_col : str, required
        The name of the column containing the score.
    match_col : str, required
        The column to use for matching control and treatment scores
        together (e.g., the size).
    match_fmt : Optional[Callable], required
        Unused.
    group_col : Optional[str], required
        The column to use as a key for coloring the points, which will
        be labeled in the legend (e.g. the treatments).
    group_fmt : Optional[Callable], required
        A function to format group labels for the legend.
    group_order : Optional[Callable], required
        A function to be called as a sort key when ordering the groups.
    subfigure_col : Optional[str], required
        The column to use to split the figure up into subfigures.
    subfigure_fmt : Optional[str], required
        A function to format the title for each subfigure.
    subfigure_order : Optional[Callable], required
        A function to be called as a sort key when ordering the
        subfigures.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray[plt.Axes]]
        A tuple containing the figure and its axes.
    """

    def plot_func(group_to_data, ax):
        for i, (group, data) in enumerate(group_to_data.items()):
            matches = data["matches"]
            control_scores = data["control_scores"]
            treatment_scores = data["treatment_scores"]

            # Fit isotonic curves for the control and treatment learning
            # curves, as well as the isotonic curves' inverses.
            min_size = matches.min()
            max_size = matches.max()

            xs = np.linspace(min_size, max_size, num=2500)

            control_smoother = IsotonicRegression(out_of_bounds="clip").fit(
                matches, control_scores
            )
            control_smoother_inv = IsotonicRegression(out_of_bounds="clip").fit(
                control_smoother.predict(xs), xs
            )

            treatment_smoother = IsotonicRegression(out_of_bounds="clip").fit(
                matches, treatment_scores
            )

            # Plot the performance equivalent curve.
            ax.plot(
                control_smoother.predict(xs),
                treatment_smoother.predict(xs),
                linestyle=LINESTYLES[i],
                label=group,
            )
            # Plot the original data points without smoothing, since
            # they're already aligned (i.e., trained on the same amount
            # of data).
            ax.scatter(
                control_scores, treatment_scores, c="k", s=8,
            )

        # Compute the minimum and maximum scores.
        min_score = min(
            min(data["control_scores"].min(), data["treatment_scores"].min())
            for data in group_to_data.values()
        )
        max_score = max(
            max(data["control_scores"].max(), data["treatment_scores"].max())
            for data in group_to_data.values()
        )
        # Scale the x and y limits
        ax.set_xlim(0.99 * min_score, 1.01 * max_score)
        ax.set_ylim(0.99 * min_score, 1.01 * max_score)
        # Plot the y = x line.
        ax.plot(
            [0.99 * min_score, 1.01 * max_score],
            [0.99 * min_score, 1.01 * max_score],
            **CENTERLINE_STYLE_KWARGS,
        )
        # Add the second axis at the top of the figure.
        def perf2cost(x):
            if len(x) == 0:
                return x
            return control_smoother_inv.predict(x.reshape(-1)).tolist()

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticklabels(
            [f"{int(x):d}" for x in perf2cost(ax2.get_xticks())]
        )

    return _make_plot_grid(
        plot_func=plot_func,
        x_label="control score (accuracy)",
        y_label="treatment score (accuracy)",
        control_data=control_data,
        treatment_data=treatment_data,
        score_col=score_col,
        match_col=match_col,
        match_fmt=None,
        group_col=group_col,
        group_fmt=group_fmt,
        group_order=group_order,
        subfigure_col=subfigure_col,
        subfigure_fmt=subfigure_fmt,
        subfigure_order=subfigure_order,
    )


# figure configuration


@dataclasses.dataclass
class FigureConfig:
    """A configuration object for a figure."""

    fig_name: str
    control_fname: str
    treatment_fname: str
    score_col: str
    hyper_param_cols: List[str]
    control_split_key: List[str]
    treatment_split_key: List[str]
    plot_func: Callable
    plot_kwargs: Dict[str, Any]


TOPIC_TO_FIGURE_CONFIG = {
    "effect-of-size": [
        FigureConfig(
            fig_name="learning-curves_compare-transfer-methods_pair-plot",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "multiset"],
            plot_func=plot_paired_performance,
            plot_kwargs={
                "match_col": "task",
                "match_fmt": lambda x: {"commonsenseqa": "CQA"}[x],
                "group_col": "transfer_method",
                "group_fmt": lambda x: {
                    "multi-task": "multitask",
                    "multi-task-fine-tune": "fine-tune",
                    "sequential-fine-tune": "sequential",
                }[x],
                "group_order": lambda x: {
                    "multi-task": 0,
                    "multi-task-fine-tune": 1,
                    "sequential-fine-tune": 2,
                }[x],
                "subfigure_col": "size",
                "subfigure_fmt": "# train examples: {:d}".format,
                "subfigure_order": int,
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-transfer-methods_cost-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["task"],
            treatment_split_key=["task", "multiset"],
            plot_func=plot_cost_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "transfer_method",
                "group_fmt": lambda x: {
                    "multi-task": "multitask",
                    "multi-task-fine-tune": "fine-tune",
                    "sequential-fine-tune": "sequential",
                }[x],
                "group_order": lambda x: {
                    "multi-task": 0,
                    "multi-task-fine-tune": 1,
                    "sequential-fine-tune": 2,
                }[x],
                "subfigure_col": "model_size",
                "subfigure_fmt": str.capitalize,
                "subfigure_order": lambda x: {
                    "small": 0,
                    "base": 1,
                    "large": 2,
                }[x],
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-transfer-methods_performance-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["task"],
            treatment_split_key=["task", "multiset"],
            plot_func=plot_performance_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "transfer_method",
                "group_fmt": lambda x: {
                    "multi-task": "multitask",
                    "multi-task-fine-tune": "fine-tune",
                    "sequential-fine-tune": "sequential",
                }[x],
                "group_order": lambda x: {
                    "multi-task": 0,
                    "multi-task-fine-tune": 1,
                    "sequential-fine-tune": 2,
                }[x],
                "subfigure_col": "model_size",
                "subfigure_fmt": str.capitalize,
                "subfigure_order": lambda x: {
                    "small": 0,
                    "base": 1,
                    "large": 2,
                }[x],
            },
        ),
    ],
    "transferring-knowledge-graphs": [
        FigureConfig(
            fig_name="full-task_compare-multisets_pair-plot",
            control_fname="single-task_full-tasks/table.csv",
            treatment_fname="multiset_full-tasks/table.csv",
            score_col="best_score",
            hyper_param_cols=["direction", "rate", "lr"],
            control_split_key=["model_size"],
            treatment_split_key=[
                "model_size",
                "knowledge-graph",
                "transfer_method",
            ],
            plot_func=plot_paired_performance,
            plot_kwargs={
                "match_col": "task",
                "match_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "knowledge-graph": "none",
                    "rainbow-knowledge-graph": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "knowledge-graph": 0,
                    "rainbow-knowledge-graph": 1,
                }[x],
                "subfigure_col": None,
                "subfigure_fmt": None,
                "subfigure_order": None,
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-multisets_pair-plot",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=[
                "model_size",
                "knowledge-graph",
                "transfer_method",
            ],
            plot_func=plot_paired_performance,
            plot_kwargs={
                "match_col": "task",
                "match_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "knowledge-graph": "none",
                    "rainbow-knowledge-graph": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "knowledge-graph": 0,
                    "rainbow-knowledge-graph": 1,
                }[x],
                "subfigure_col": "size",
                "subfigure_fmt": "# train examples: {:d}".format,
                "subfigure_order": int,
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-multisets_cost-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=[
                "model_size",
                "knowledge-graph",
                "transfer_method",
            ],
            plot_func=plot_cost_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "knowledge-graph": "none",
                    "rainbow-knowledge-graph": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "knowledge-graph": 0,
                    "rainbow-knowledge-graph": 1,
                }[x],
                "subfigure_col": "task",
                "subfigure_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "subfigure_order": lambda x: {
                    "anli": 0,
                    "cosmosqa": 1,
                    "hellaswag": 2,
                    "physicaliqa": 3,
                    "socialiqa": 4,
                    "winogrande": 5,
                }[x],
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-multisets_performance-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=[
                "model_size",
                "knowledge-graph",
                "transfer_method",
            ],
            plot_func=plot_performance_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "knowledge-graph": "none",
                    "rainbow-knowledge-graph": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "knowledge-graph": 0,
                    "rainbow-knowledge-graph": 1,
                }[x],
                "subfigure_col": "task",
                "subfigure_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "subfigure_order": lambda x: {
                    "anli": 0,
                    "cosmosqa": 1,
                    "hellaswag": 2,
                    "physicaliqa": 3,
                    "socialiqa": 4,
                    "winogrande": 5,
                }[x],
            },
        ),
        FigureConfig(
            fig_name="full-task_compare-knowledge-graphs_pair-plot",
            control_fname="single-task_full-tasks/table.csv",
            treatment_fname="multiset_full-tasks/table.csv",
            score_col="best_score",
            hyper_param_cols=["direction", "rate", "lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "multiset", "transfer_method"],
            plot_func=plot_paired_performance,
            plot_kwargs={
                "match_col": "task",
                "match_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "group_col": "knowledge-graph",
                "group_fmt": lambda x: {
                    "atomic": "ATOMIC",
                    "conceptnet": "ConceptNet",
                    "comet": "Both",
                }[x],
                "group_order": lambda x: {
                    "atomic": 0,
                    "conceptnet": 1,
                    "comet": 2,
                }[x],
                "subfigure_col": None,
                "subfigure_fmt": None,
                "subfigure_order": None,
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-knowledge-graphs_pair-plot",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "multiset", "transfer_method"],
            plot_func=plot_paired_performance,
            plot_kwargs={
                "match_col": "task",
                "match_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "group_col": "knowledge-graph",
                "group_fmt": lambda x: {
                    "atomic": "ATOMIC",
                    "conceptnet": "ConceptNet",
                    "comet": "Both",
                }[x],
                "group_order": lambda x: {
                    "atomic": 0,
                    "conceptnet": 1,
                    "comet": 2,
                }[x],
                "subfigure_col": "size",
                "subfigure_fmt": "# train examples: {:d}".format,
                "subfigure_order": int,
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-knowledge-graphs_cost-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "multiset", "transfer_method"],
            plot_func=plot_cost_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "knowledge-graph",
                "group_fmt": lambda x: {
                    "atomic": "ATOMIC",
                    "conceptnet": "ConceptNet",
                    "comet": "Both",
                }[x],
                "group_order": lambda x: {
                    "atomic": 0,
                    "conceptnet": 1,
                    "comet": 2,
                }[x],
                "subfigure_col": "task",
                "subfigure_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "subfigure_order": lambda x: {
                    "anli": 0,
                    "cosmosqa": 1,
                    "hellaswag": 2,
                    "physicaliqa": 3,
                    "socialiqa": 4,
                    "winogrande": 5,
                }[x],
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-knowledge-graphs_performance-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "multiset", "transfer_method"],
            plot_func=plot_performance_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "knowledge-graph",
                "group_fmt": lambda x: {
                    "atomic": "ATOMIC",
                    "conceptnet": "ConceptNet",
                    "comet": "Both",
                }[x],
                "group_order": lambda x: {
                    "atomic": 0,
                    "conceptnet": 1,
                    "comet": 2,
                }[x],
                "subfigure_col": "task",
                "subfigure_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "subfigure_order": lambda x: {
                    "anli": 0,
                    "cosmosqa": 1,
                    "hellaswag": 2,
                    "physicaliqa": 3,
                    "socialiqa": 4,
                    "winogrande": 5,
                }[x],
            },
        ),
    ],
    "transferring-multisets": [
        FigureConfig(
            fig_name="full-task_compare-multisets_pair-plot",
            control_fname="single-task_full-tasks/table.csv",
            treatment_fname="multiset_full-tasks/table.csv",
            score_col="best_score",
            hyper_param_cols=["rate", "lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "transfer_method"],
            plot_func=plot_paired_performance,
            plot_kwargs={
                "match_col": "task",
                "match_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "glue": "GLUE",
                    "super-glue": "SuperGLUE",
                    "rainbow": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "rainbow": 0,
                    "glue": 1,
                    "super-glue": 2,
                }[x],
                "subfigure_col": None,
                "subfigure_fmt": None,
                "subfigure_order": None,
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-multisets_pair-plot",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "transfer_method"],
            plot_func=plot_paired_performance,
            plot_kwargs={
                "match_col": "task",
                "match_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "glue": "GLUE",
                    "super-glue": "SuperGLUE",
                    "rainbow": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "rainbow": 0,
                    "glue": 1,
                    "super-glue": 2,
                }[x],
                "subfigure_col": "size",
                "subfigure_fmt": "# train examples: {:d}".format,
                "subfigure_order": int,
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-multisets_cost-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "transfer_method"],
            plot_func=plot_cost_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "glue": "GLUE",
                    "super-glue": "SuperGLUE",
                    "rainbow": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "rainbow": 0,
                    "glue": 1,
                    "super-glue": 2,
                }[x],
                "subfigure_col": "task",
                "subfigure_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "subfigure_order": lambda x: {
                    "anli": 0,
                    "cosmosqa": 1,
                    "hellaswag": 2,
                    "physicaliqa": 3,
                    "socialiqa": 4,
                    "winogrande": 5,
                }[x],
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-multisets_performance-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "transfer_method"],
            plot_func=plot_performance_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "glue": "GLUE",
                    "super-glue": "SuperGLUE",
                    "rainbow": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "rainbow": 0,
                    "glue": 1,
                    "super-glue": 2,
                }[x],
                "subfigure_col": "task",
                "subfigure_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "subfigure_order": lambda x: {
                    "anli": 0,
                    "cosmosqa": 1,
                    "hellaswag": 2,
                    "physicaliqa": 3,
                    "socialiqa": 4,
                    "winogrande": 5,
                }[x],
            },
        ),
        FigureConfig(
            fig_name="full-task_compare-transfer-methods_pair-plot",
            control_fname="single-task_full-tasks/table.csv",
            treatment_fname="multiset_full-tasks/table.csv",
            score_col="best_score",
            hyper_param_cols=["rate", "lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "multiset"],
            plot_func=plot_paired_performance,
            plot_kwargs={
                "match_col": "task",
                "match_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "group_col": "transfer_method",
                "group_fmt": lambda x: {
                    "multi-task": "multitask",
                    "multi-task-fine-tune": "fine-tune",
                    "sequential-fine-tune": "sequential",
                }[x],
                "group_order": lambda x: {
                    "multi-task": 0,
                    "multi-task-fine-tune": 1,
                    "sequential-fine-tune": 2,
                }[x],
                "subfigure_col": None,
                "subfigure_fmt": None,
                "subfigure_order": None,
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-transfer-methods_pair-plot",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "multiset"],
            plot_func=plot_paired_performance,
            plot_kwargs={
                "match_col": "task",
                "match_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "group_col": "transfer_method",
                "group_fmt": lambda x: {
                    "multi-task": "multitask",
                    "multi-task-fine-tune": "fine-tune",
                    "sequential-fine-tune": "sequential",
                }[x],
                "group_order": lambda x: {
                    "multi-task": 0,
                    "multi-task-fine-tune": 1,
                    "sequential-fine-tune": 2,
                }[x],
                "subfigure_col": "size",
                "subfigure_fmt": "# train examples: {:d}".format,
                "subfigure_order": int,
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-transfer-methods_cost-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "multiset"],
            plot_func=plot_cost_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "transfer_method",
                "group_fmt": lambda x: {
                    "multi-task": "multitask",
                    "multi-task-fine-tune": "fine-tune",
                    "sequential-fine-tune": "sequential",
                }[x],
                "group_order": lambda x: {
                    "multi-task": 0,
                    "multi-task-fine-tune": 1,
                    "sequential-fine-tune": 2,
                }[x],
                "subfigure_col": "task",
                "subfigure_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "subfigure_order": lambda x: {
                    "anli": 0,
                    "cosmosqa": 1,
                    "hellaswag": 2,
                    "physicaliqa": 3,
                    "socialiqa": 4,
                    "winogrande": 5,
                }[x],
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-transfer-methods_performance-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "multiset"],
            plot_func=plot_performance_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "transfer_method",
                "group_fmt": lambda x: {
                    "multi-task": "multitask",
                    "multi-task-fine-tune": "fine-tune",
                    "sequential-fine-tune": "sequential",
                }[x],
                "group_order": lambda x: {
                    "multi-task": 0,
                    "multi-task-fine-tune": 1,
                    "sequential-fine-tune": 2,
                }[x],
                "subfigure_col": "task",
                "subfigure_fmt": lambda x: {
                    "anli": "ART",
                    "cosmosqa": "CosmosQA",
                    "hellaswag": "HSWAG",
                    "physicaliqa": "PIQA",
                    "socialiqa": "SocialIQa",
                    "winogrande": "WinoGrande",
                }[x],
                "subfigure_order": lambda x: {
                    "anli": 0,
                    "cosmosqa": 1,
                    "hellaswag": 2,
                    "physicaliqa": 3,
                    "socialiqa": 4,
                    "winogrande": 5,
                }[x],
            },
        ),
    ],
    "transferring-to-external-tasks": [
        FigureConfig(
            fig_name="full-task_compare-multisets_pair-plot",
            control_fname="single-task_full-tasks/table.csv",
            treatment_fname="multiset_full-tasks/table.csv",
            score_col="best_score",
            hyper_param_cols=["rate", "lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "transfer_method"],
            plot_func=plot_paired_performance,
            plot_kwargs={
                "match_col": "task",
                "match_fmt": lambda x: {
                    "commonsenseqa": "CQA",
                    "joci": "JOCI",
                }[x],
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "glue": "GLUE",
                    "super-glue": "SuperGLUE",
                    "rainbow": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "rainbow": 0,
                    "glue": 1,
                    "super-glue": 2,
                }[x],
                "subfigure_col": None,
                "subfigure_fmt": None,
                "subfigure_order": None,
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-multisets_pair-plot",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "transfer_method"],
            plot_func=plot_paired_performance,
            plot_kwargs={
                "match_col": "task",
                "match_fmt": lambda x: {
                    "commonsenseqa": "CQA",
                    "joci": "JOCI",
                }[x],
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "glue": "GLUE",
                    "super-glue": "SuperGLUE",
                    "rainbow": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "rainbow": 0,
                    "glue": 1,
                    "super-glue": 2,
                }[x],
                "subfigure_col": "size",
                "subfigure_fmt": "# train examples: {:d}".format,
                "subfigure_order": int,
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-multisets_cost-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "transfer_method"],
            plot_func=plot_cost_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "glue": "GLUE",
                    "super-glue": "SuperGLUE",
                    "rainbow": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "rainbow": 0,
                    "glue": 1,
                    "super-glue": 2,
                }[x],
                "subfigure_col": "task",
                "subfigure_fmt": lambda x: {
                    "commonsenseqa": "CommonsenseQA",
                    "joci": "JOCI",
                }[x],
                "subfigure_order": lambda x: {"commonsenseqa": 0, "joci": 1}[x],
            },
        ),
        FigureConfig(
            fig_name="learning-curves_compare-multisets_performance-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size"],
            treatment_split_key=["model_size", "transfer_method"],
            plot_func=plot_performance_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "glue": "GLUE",
                    "super-glue": "SuperGLUE",
                    "rainbow": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "rainbow": 0,
                    "glue": 1,
                    "super-glue": 2,
                }[x],
                "subfigure_col": "task",
                "subfigure_fmt": lambda x: {
                    "commonsenseqa": "CommonsenseQA",
                    "joci": "JOCI",
                }[x],
                "subfigure_order": lambda x: {"commonsenseqa": 0, "joci": 1}[x],
            },
        ),
        # Make equivalent curves for individual tasks to use in
        # illustrating how equivalent curves work.
        FigureConfig(
            fig_name="learning-curves_task_cost-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size", "task"],
            treatment_split_key=["model_size", "task", "transfer_method"],
            plot_func=plot_cost_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "glue": "GLUE",
                    "super-glue": "SuperGLUE",
                    "rainbow": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "rainbow": 0,
                    "glue": 1,
                    "super-glue": 2,
                }[x],
                "subfigure_col": "task",
                "subfigure_fmt": lambda x: {
                    "commonsenseqa": "CommonsenseQA",
                    "joci": "JOCI",
                }[x],
                "subfigure_order": None,
            },
        ),
        FigureConfig(
            fig_name="learning-curves_task_performance-equivalent-curve",
            control_fname="single-task_learning-curves/table.csv",
            treatment_fname="multiset_learning-curves/table.csv",
            score_col="best_score",
            hyper_param_cols=["lr"],
            control_split_key=["model_size", "task"],
            treatment_split_key=["model_size", "task", "transfer_method"],
            plot_func=plot_performance_equivalent_curves,
            plot_kwargs={
                "match_col": "size",
                "match_fmt": None,
                "group_col": "multiset",
                "group_fmt": lambda x: {
                    "glue": "GLUE",
                    "super-glue": "SuperGLUE",
                    "rainbow": "Rainbow",
                }[x],
                "group_order": lambda x: {
                    "rainbow": 0,
                    "glue": 1,
                    "super-glue": 2,
                }[x],
                "subfigure_col": "task",
                "subfigure_fmt": lambda x: {
                    "commonsenseqa": "CommonsenseQA",
                    "joci": "JOCI",
                }[x],
                "subfigure_order": None,
            },
        ),
    ],
}
"""Figure configurations for all experiments."""


# main function


@click.command()
@click.argument(
    "src", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "dst", type=click.Path(exists=False, dir_okay=True, file_okay=False)
)
def create_multi_experiment_figures(src: str, dst: str) -> None:
    """Create multi-experiment figures for the Rainbow results.

    Read in the raw tables from SRC and write out the figures to DST.
    """
    utils.configure_logging(clear=True)

    for topic, figure_configs in tqdm.tqdm(
        TOPIC_TO_FIGURE_CONFIG.items(), **settings.TQDM_KWARGS
    ):
        for config in tqdm.tqdm(figure_configs, **settings.TQDM_KWARGS):
            os.makedirs(os.path.join(dst, topic, config.fig_name))
            # Read in the data.
            control_fpath = os.path.join(src, topic, config.control_fname)
            control_data = (
                pd.read_csv(control_fpath)
                if control_fpath.endswith("csv")
                else pd.read_json(control_fpath, lines=True)
            )
            treatment_fpath = os.path.join(src, topic, config.treatment_fname)
            treatment_data = (
                pd.read_csv(treatment_fpath)
                if treatment_fpath.endswith("csv")
                else pd.read_json(treatment_fpath, lines=True)
            )
            # Max over the hyper-parameters.
            treatment_data = (
                treatment_data.groupby(
                    [
                        col
                        for col in treatment_data.columns
                        if col not in config.hyper_param_cols
                        and col != config.score_col
                    ]
                )
                .max()[config.score_col]
                .reset_index()
            )
            control_data = (
                control_data.groupby(
                    [
                        col
                        for col in control_data.columns
                        if col not in config.hyper_param_cols
                        and col != config.score_col
                    ]
                )
                .max()[config.score_col]
                .reset_index()
            )
            for key, treatment_subdata in treatment_data.groupby(
                config.treatment_split_key
            ):
                control_subdata = control_data[
                    # Select only rows which agree with the current key.
                    functools.reduce(
                        operator.and_,
                        [
                            control_data[key_name] == key_value
                            for key_name, key_value in zip(
                                config.treatment_split_key, key
                            )
                            if key_name in config.control_split_key
                        ],
                    )
                ]

                fig, axes = config.plot_func(
                    control_data=control_subdata,
                    treatment_data=treatment_subdata,
                    score_col=config.score_col,
                    **config.plot_kwargs,
                )
                dst_path = os.path.join(
                    dst,
                    topic,
                    config.fig_name,
                    ".".join(list(key) + [config.fig_name, "pdf"]),
                )
                fig.savefig(dst_path)
                plt.close(fig)


if __name__ == "__main__":
    create_multi_experiment_figures()  # pylint: disable=no-value-for-parameter
