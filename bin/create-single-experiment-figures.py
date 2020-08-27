#! /usr/bin/env python

"""Create single experiment figures for the Rainbow results."""

import dataclasses
import logging
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


# helper functions


def plot_training_curves(
    data: pd.DataFrame,
    training_curve: str,
    group: str,
    group_fmt: Optional[Callable] = None,
    fig_row: Optional[str] = None,
    fig_row_fmt: Optional[Callable] = None,
    fig_col: Optional[str] = None,
    fig_col_fmt: Optional[Callable] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Return a plot of the training curves.

    Parameters
    ----------
    data : pd.DataFrame, required
        The data to use for the plot.
    training_curve : str, required
        The name of the column containing the training curves as lists
        of ``[step, score]`` pairs.
    group : str, required
        The column to use as a key for plotting multiple curves at a
        time. The column's value will be used in the legend to identify
        the curves.
    group_fmt : Optional[Callable], optional (default=None)
        A function to format group labels for the legend.
    fig_row : Optional[str], optional (default=None)
        An optional column to use for determining the rows of subfigures
        that the curves will then reside in.
    fig_row_fmt : Optional[Callable], optional (default=None)
        A function to format figure row labels.
    fig_col : Optional[str], optional (default=None)
        An optional column to use for determining the columns of
        subfigures that the curves will then reside in.
    fig_col_fmt : Optional[Callable], optional (default=None)
        A function to format figure column labels.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray[plt.Axes]]
        A tuple containing the figure and its axes.
    """
    group_fmt = group_fmt if group_fmt is not None else lambda x: x
    fig_row_fmt = fig_row_fmt if fig_row_fmt is not None else lambda x: x
    fig_col_fmt = fig_col_fmt if fig_col_fmt is not None else lambda x: x

    row_labels = (
        sorted(data[fig_row].unique()) if fig_row is not None else [None]
    )
    col_labels = (
        sorted(data[fig_col].unique()) if fig_col is not None else [None]
    )

    n_rows = len(row_labels)
    n_cols = len(col_labels)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(6 * n_cols, 6 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    # Modify axes so we can access the axis objects in a uniform way,
    # regardless of the number of rows or columns.
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(n_rows, n_cols)

    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            subdata = data[
                (
                    data[fig_row] == row_label
                    if fig_row is not None
                    else np.repeat(True, len(data))
                )
                & (
                    data[fig_col] == col_label
                    if fig_col is not None
                    else np.repeat(True, len(data))
                )
            ]
            group_labels = sorted(subdata[group].unique())
            for group_label in group_labels:
                curve_data = subdata[subdata[group] == group_label][
                    training_curve
                ]
                if len(curve_data) != 1:
                    raise ValueError(
                        f"Found multiple rows corresponding to {row_label},"
                        f" {col_label}, {group_label}."
                    )
                curve_data = curve_data.iloc[0]

                axes[row_idx][col_idx].plot(
                    *zip(*curve_data), label=group_fmt(group_label)
                )
            axes[row_idx][col_idx].legend()

    for row_idx, row_label in enumerate(row_labels):
        # Set the label for the row.
        ax2 = axes[row_idx][-1].twinx()
        ax2.set_ylabel(fig_row_fmt(row_label), rotation=270, va="bottom")
        ax2.set_yticks([])
        # Set the label for the y-axis.
        axes[row_idx][0].set_ylabel("accuracy")

    for col_idx, col_label in enumerate(col_labels):
        # Set the label for the column.
        axes[0][col_idx].set_title(fig_col_fmt(col_label))
        # Set the label for the x-axis.
        axes[-1][col_idx].set_xlabel("step")

    return fig, axes


def plot_learning_curves(
    data: pd.DataFrame,
    training_curve: str,
    dataset_size: str,
    group: str,
    group_fmt: Optional[Callable] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Return a plot of the learning curves.

    Parameters
    ----------
    data : pd.DataFrame, required
        The data to use for the plot.
    training_curve : str, required
        The name of the column containing the training curves as lists
        of ``[step, score]`` pairs.
    dataset_size : str, required
        The name of the column containing the dataset size used for
        training the model represented by that row.
    group : str, required
        The column to use as a key for plotting multiple curves at a
        time. The column's value will be used in the legend to identify
        the curves.
    group_fmt : Optional[Callable], optional (default=None)
        A function to format group labels for the legend.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray[plt.Axes]]
        A tuple containing the figure and its axes.
    """
    group_fmt = group_fmt if group_fmt is not None else lambda x: x

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(11, 11), sharey=True, constrained_layout=True
    )

    sizes = sorted(data[dataset_size].unique())

    group_labels = sorted(data[group].unique())

    # Check that the data frame only has one row for each unique value
    # of the "group" variable.
    n_group = len(group_labels)
    group_sizes = data.groupby(dataset_size).count()[group].unique()
    if len(group_sizes) != 1 or group_sizes[0] != n_group:
        raise ValueError(
            f"data has the wrong number of rows for each dataset size."
            f" There should be exactly {n_group} rows for each value of"
            f" {dataset_size}: one row for each possible value of"
            f" {group}."
        )

    # Check that each group has the correct number of dataset sizes.
    n_sizes = len(sizes)
    n_sizes_per_group = data.groupby(group).count()[dataset_size].unique()
    if len(n_sizes_per_group) != 1 or n_sizes_per_group[0] != n_sizes:
        raise ValueError(
            f"data has the wrong number of rows for each value of"
            f" {group}. There should be exactly {n_sizes} rows for each"
            f" value, one for every dataset size."
        )

    # Compute the best score from the training curve.
    data = data.copy()
    data["_best_score"] = data[training_curve].apply(
        lambda curve: max(score for _, score in curve)
    )

    # Plot the learning curve for each learning rate.
    for group_label in group_labels:
        learning_curve = data[data[group] == group_label][
            [dataset_size, "_best_score"]
        ]
        axes[0][0].plot(
            learning_curve[dataset_size],
            learning_curve["_best_score"],
            label=group_fmt(group_label),
        )
        axes[1][0].plot(
            learning_curve[dataset_size],
            learning_curve["_best_score"],
            label=group_fmt(group_label),
        )

    axes[0][0].legend()
    axes[1][0].legend()

    axes[1][0].set_xscale("log")

    axes[0][0].set_title("Learning Curve by Learning Rate")

    # Plot the profile learning curve and a smoothed version using
    # isotonic regression.
    profile_learning_curve = data.groupby(dataset_size).max()["_best_score"]
    xs = profile_learning_curve.index
    ys = profile_learning_curve.values

    smoother = IsotonicRegression(out_of_bounds="clip")

    axes[0][1].plot(xs, ys, label="original", c="b", marker="o")
    axes[0][1].plot(xs, smoother.fit_transform(xs, ys), label="smoothed", c="r")
    axes[1][1].plot(xs, ys, label="original", c="b", marker="o")
    axes[1][1].plot(xs, smoother.fit_transform(xs, ys), label="smoothed", c="r")

    axes[0][1].legend()
    axes[1][1].legend()

    axes[1][1].set_xscale("log")

    axes[0][1].set_title("Profile Learning Curve")

    for i in range(2):
        # Set the label for the y-axis.
        axes[i][0].set_ylabel("accuracy")

    for i in range(2):
        # Set the label for the x-axis.
        axes[-1][i].set_xlabel("# examples")

    return fig, axes


# figure configuration


@dataclasses.dataclass
class FigureConfig:
    """A configuration object for a figure."""

    fig_name: str
    data_fname: str
    split_key: List[str]
    plot_func: Callable
    plot_kwargs: Dict[str, Any]


TOPIC_TO_FIGURE_CONFIG = {
    "effect-of-size": {
        "mixtures": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset", "rate"],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": None,
                    "fig_row_fmt": None,
                    "fig_col": None,
                    "fig_col_fmt": None,
                },
            ),
        ],
        "multiset_learning-curves": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset", "transfer_method"],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": "size",
                    "fig_row_fmt": "size: {:d}".format,
                    "fig_col": None,
                    "fig_col_fmt": None,
                },
            ),
            FigureConfig(
                fig_name="learning-curves",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset", "transfer_method"],
                plot_func=plot_learning_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "dataset_size": "size",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                },
            ),
        ],
        "single-task_learning-curves": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset"],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": "size",
                    "fig_row_fmt": "size: {:d}".format,
                    "fig_col": None,
                    "fig_col_fmt": None,
                },
            ),
            FigureConfig(
                fig_name="learning-curves",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset"],
                plot_func=plot_learning_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "dataset_size": "size",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                },
            ),
        ],
    },
    "transferring-knowledge-graphs": {
        "multiset_full-tasks": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=[
                    "model_size",
                    "task",
                    "multiset",
                    "knowledge-graph",
                    "transfer_method",
                ],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": "rate",
                    "fig_row_fmt": None,
                    "fig_col": "direction",
                    "fig_col_fmt": None,
                },
            ),
            FigureConfig(
                fig_name="direction-comparison",
                data_fname="training-curves.jsonl",
                split_key=[
                    "model_size",
                    "task",
                    "multiset",
                    "knowledge-graph",
                    "transfer_method",
                ],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "direction",
                    "group_fmt": None,
                    "fig_row": "rate",
                    "fig_row_fmt": None,
                    "fig_col": "lr",
                    "fig_col_fmt": "lr: {:.1e}".format,
                },
            ),
        ],
        "multiset_learning-curves": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=[
                    "model_size",
                    "task",
                    "multiset",
                    "knowledge-graph",
                    "transfer_method",
                ],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": "size",
                    "fig_row_fmt": "size: {:d}".format,
                    "fig_col": None,
                    "fig_col_fmt": None,
                },
            ),
            FigureConfig(
                fig_name="learning-curves",
                data_fname="training-curves.jsonl",
                split_key=[
                    "model_size",
                    "task",
                    "multiset",
                    "knowledge-graph",
                    "transfer_method",
                ],
                plot_func=plot_learning_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "dataset_size": "size",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                },
            ),
        ],
        "single-task_full-tasks": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset"],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": None,
                    "fig_row_fmt": None,
                    "fig_col": None,
                    "fig_col_fmt": None,
                },
            ),
        ],
        "single-task_learning-curves": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset"],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": "size",
                    "fig_row_fmt": "size: {:d}".format,
                    "fig_col": None,
                    "fig_col_fmt": None,
                },
            ),
            FigureConfig(
                fig_name="learning-curves",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset"],
                plot_func=plot_learning_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "dataset_size": "size",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                },
            ),
        ],
    },
    "transferring-multisets": {
        "multiset_full-tasks": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset", "transfer_method"],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": None,
                    "fig_row_fmt": None,
                    "fig_col": "rate",
                    "fig_col_fmt": None,
                },
            ),
        ],
        "multiset_learning-curves": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset", "transfer_method"],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": "size",
                    "fig_row_fmt": "size: {:d}".format,
                    "fig_col": None,
                    "fig_col_fmt": None,
                },
            ),
            FigureConfig(
                fig_name="learning-curves",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset", "transfer_method"],
                plot_func=plot_learning_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "dataset_size": "size",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                },
            ),
        ],
        "single-task_full-tasks": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset"],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": None,
                    "fig_row_fmt": None,
                    "fig_col": None,
                    "fig_col_fmt": None,
                },
            ),
        ],
        "single-task_learning-curves": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset"],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": "size",
                    "fig_row_fmt": "size: {:d}".format,
                    "fig_col": None,
                    "fig_col_fmt": None,
                },
            ),
            FigureConfig(
                fig_name="learning-curves",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset"],
                plot_func=plot_learning_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "dataset_size": "size",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                },
            ),
        ],
    },
    "transferring-to-external-tasks": {
        "multiset_full-tasks": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset", "transfer_method"],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": None,
                    "fig_row_fmt": None,
                    "fig_col": "rate",
                    "fig_col_fmt": None,
                },
            ),
        ],
        "multiset_learning-curves": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset", "transfer_method"],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": "size",
                    "fig_row_fmt": "size: {:d}".format,
                    "fig_col": None,
                    "fig_col_fmt": None,
                },
            ),
            FigureConfig(
                fig_name="learning-curves",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset", "transfer_method"],
                plot_func=plot_learning_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "dataset_size": "size",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                },
            ),
        ],
        "single-task_full-tasks": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset"],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": None,
                    "fig_row_fmt": None,
                    "fig_col": None,
                    "fig_col_fmt": None,
                },
            ),
        ],
        "single-task_learning-curves": [
            FigureConfig(
                fig_name="learning-rate-comparison",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset"],
                plot_func=plot_training_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                    "fig_row": "size",
                    "fig_row_fmt": "size: {:d}".format,
                    "fig_col": None,
                    "fig_col_fmt": None,
                },
            ),
            FigureConfig(
                fig_name="learning-curves",
                data_fname="training-curves.jsonl",
                split_key=["model_size", "task", "multiset"],
                plot_func=plot_learning_curves,
                plot_kwargs={
                    "training_curve": "training_curve",
                    "dataset_size": "size",
                    "group": "lr",
                    "group_fmt": "lr: {:.1e}".format,
                },
            ),
        ],
    },
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
def create_single_experiment_figures(src: str, dst: str) -> None:
    """Create single experiment figures for the Rainbow results.

    Read in the raw tables from SRC and write out the figures to DST.
    """
    utils.configure_logging(clear=True)

    for topic, experiment_to_figure_configs in tqdm.tqdm(
        TOPIC_TO_FIGURE_CONFIG.items(), **settings.TQDM_KWARGS
    ):
        for experiment, figure_configs in tqdm.tqdm(
            experiment_to_figure_configs.items(), **settings.TQDM_KWARGS
        ):
            for config in tqdm.tqdm(figure_configs, **settings.TQDM_KWARGS):
                os.makedirs(
                    os.path.join(dst, topic, experiment, config.fig_name)
                )

                src_path = os.path.join(
                    src, topic, experiment, config.data_fname
                )
                data = (
                    pd.read_csv(src_path)
                    if src_path.endswith("csv")
                    else pd.read_json(src_path, lines=True)
                )
                for key, subdata in data.groupby(config.split_key):
                    fig, axes = config.plot_func(
                        data=subdata, **config.plot_kwargs
                    )

                    dst_path = os.path.join(
                        dst,
                        topic,
                        experiment,
                        config.fig_name,
                        ".".join(list(key) + [config.fig_name, "pdf"]),
                    )
                    fig.savefig(dst_path)
                    plt.close(fig)


if __name__ == "__main__":
    create_single_experiment_figures()  # pylint: disable=no-value-for-parameter
