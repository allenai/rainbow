#! /usr/bin/env python

"""Generate tables from Rainbow's experimental results."""

import csv
import json
import logging
import os
import re
from typing import Any, List, Optional, Tuple

import click
from sklearn import metrics

from rainbow import utils


logger = logging.getLogger(__name__)


# constants

EXPERIMENT_TO_TABLES_CONFIG = {
    "effect-of-size": {
        # the mixtures table
        "mixtures": (
            # path to root
            "mixtures/t5",
            # experiment factors
            {
                "model_size": ["small", "base", "large"],
                "task": [
                    "anli",
                    "cosmosqa",
                    "hellaswag",
                    "physicaliqa",
                    "socialiqa",
                    "winogrande",
                ],
                "multiset": ["rainbow"],
                "rate": ["equal"],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            [],
        ),
        # the multiset learning curves table
        "multiset_learning-curves": (
            # path to root
            "learning-curves/t5",
            # experiment factors
            {
                "model_size": ["small", "base", "large"],
                "task": ["commonsenseqa"],
                "multiset": ["rainbow"],
                "transfer_method": [
                    "multi-task",
                    "multi-task-fine-tune",
                    "sequential-fine-tune",
                ],
                "size": [
                    "00004",
                    "00010",
                    "00030",
                    "00091",
                    "00280",
                    "00865",
                    "02667",
                    "05334",
                    "08000",
                    "10667",
                    "13334",
                    "16000",
                ],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["single-task"],
        ),
        # the single-task learning curves table
        "single-task_learning-curves": (
            # path to root
            "learning-curves/t5",
            # experiment factors
            {
                "model_size": ["small", "base", "large"],
                "task": ["commonsenseqa"],
                "multiset": ["single-task"],
                "size": [
                    "00004",
                    "00010",
                    "00030",
                    "00091",
                    "00280",
                    "00865",
                    "02667",
                    "05334",
                    "08000",
                    "10667",
                    "13334",
                    "16000",
                ],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["rainbow"],
        ),
    },
    "transferring-multisets": {
        # the multiset full tasks table
        "multiset_full-tasks": (
            # path to root
            "full-tasks/t5",
            # experiment factors
            {
                "model_size": ["large"],
                "task": [
                    "anli",
                    "cosmosqa",
                    "hellaswag",
                    "physicaliqa",
                    "socialiqa",
                    "winogrande",
                ],
                "multiset": ["glue", "super-glue", "rainbow"],
                "transfer_method": [
                    "multi-task",
                    "multi-task-fine-tune",
                    "sequential-fine-tune",
                ],
                "rate": ["equal", "proportional"],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["single-task"],
        ),
        # the single-task full tasks table
        "single-task_full-tasks": (
            # path to root
            "full-tasks/t5",
            # experiment factors
            {
                "model_size": ["large"],
                "task": [
                    "anli",
                    "cosmosqa",
                    "hellaswag",
                    "physicaliqa",
                    "socialiqa",
                    "winogrande",
                ],
                "multiset": ["single-task"],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["glue", "super-glue", "rainbow"],
        ),
        # the multiset learning curves table
        "multiset_learning-curves": (
            # path to root
            "learning-curves/t5",
            # experiment factors
            {
                "model_size": ["large"],
                "task": [
                    "anli",
                    "cosmosqa",
                    "hellaswag",
                    "physicaliqa",
                    "socialiqa",
                    "winogrande",
                ],
                "multiset": ["glue", "super-glue", "rainbow"],
                "transfer_method": [
                    "multi-task",
                    "multi-task-fine-tune",
                    "sequential-fine-tune",
                ],
                "size": [
                    "00004",
                    "00010",
                    "00030",
                    "00091",
                    "00280",
                    "00865",
                    "02667",
                    "05334",
                    "08000",
                    "10667",
                    "13334",
                    "16000",
                ],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["single-task"],
        ),
        # the single-task learning curves table
        "single-task_learning-curves": (
            # path to root
            "learning-curves/t5",
            # experiment factors
            {
                "model_size": ["large"],
                "task": [
                    "anli",
                    "cosmosqa",
                    "hellaswag",
                    "physicaliqa",
                    "socialiqa",
                    "winogrande",
                ],
                "multiset": ["single-task"],
                "size": [
                    "00004",
                    "00010",
                    "00030",
                    "00091",
                    "00280",
                    "00865",
                    "02667",
                    "05334",
                    "08000",
                    "10667",
                    "13334",
                    "16000",
                ],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["glue", "super-glue", "rainbow"],
        ),
    },
    "transferring-knowledge-graphs": {
        # the multiset full tasks table
        "multiset_full-tasks": (
            # path to root
            "full-tasks/t5",
            # experiment factors
            {
                "model_size": ["large"],
                "task": [
                    "anli",
                    "cosmosqa",
                    "hellaswag",
                    "physicaliqa",
                    "socialiqa",
                    "winogrande",
                ],
                "multiset": ["knowledge-graph", "rainbow-knowledge-graph"],
                "knowledge-graph": ["atomic", "conceptnet", "comet"],
                "direction": ["forward", "backward", "bidirectional"],
                "transfer_method": ["multi-task"],
                "rate": ["equal", "proportional"],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["single-task"],
        ),
        # the single-task full tasks table
        "single-task_full-tasks": (
            # path to root
            "full-tasks/t5",
            # experiment factors
            {
                "model_size": ["large"],
                "task": [
                    "anli",
                    "cosmosqa",
                    "hellaswag",
                    "physicaliqa",
                    "socialiqa",
                    "winogrande",
                ],
                "multiset": ["single-task"],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["knowledge-graph", "rainbow-knowledge-graph"],
        ),
        # the multiset learning curves table
        "multiset_learning-curves": (
            # path to root
            "learning-curves/t5",
            # experiment factors
            {
                "model_size": ["large"],
                "task": [
                    "anli",
                    "cosmosqa",
                    "hellaswag",
                    "physicaliqa",
                    "socialiqa",
                    "winogrande",
                ],
                "multiset": ["knowledge-graph", "rainbow-knowledge-graph"],
                "knowledge-graph": ["atomic", "conceptnet", "comet"],
                "transfer_method": ["multi-task"],
                "size": [
                    "00004",
                    "00010",
                    "00030",
                    "00091",
                    "00280",
                    "00865",
                    "02667",
                    "05334",
                    "08000",
                    "10667",
                    "13334",
                    "16000",
                ],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["single-task"],
        ),
        # the single-task learning curves table
        "single-task_learning-curves": (
            # path to root
            "learning-curves/t5",
            # experiment factors
            {
                "model_size": ["large"],
                "task": [
                    "anli",
                    "cosmosqa",
                    "hellaswag",
                    "physicaliqa",
                    "socialiqa",
                    "winogrande",
                ],
                "multiset": ["single-task"],
                "size": [
                    "00004",
                    "00010",
                    "00030",
                    "00091",
                    "00280",
                    "00865",
                    "02667",
                    "05334",
                    "08000",
                    "10667",
                    "13334",
                    "16000",
                ],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["knowledge-graph", "rainbow-knowledge-graph"],
        ),
    },
    "transferring-to-external-tasks": {
        # the multiset full tasks table
        "multiset_full-tasks": (
            # path to root
            "full-tasks/t5",
            # experiment factors
            {
                "model_size": ["large"],
                "task": ["commonsenseqa", "joci"],
                "multiset": ["glue", "super-glue", "rainbow"],
                "transfer_method": ["multi-task"],
                "rate": ["equal", "proportional"],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["single-task"],
        ),
        # the single-task full tasks table
        "single-task_full-tasks": (
            # path to root
            "full-tasks/t5",
            # experiment factors
            {
                "model_size": ["large"],
                "task": ["commonsenseqa", "joci"],
                "multiset": ["single-task"],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["glue", "super-glue", "rainbow"],
        ),
        # the multiset learning curves table
        "multiset_learning-curves": (
            # path to root
            "learning-curves/t5",
            # experiment factors
            {
                "model_size": ["large"],
                "task": ["commonsenseqa", "joci"],
                "multiset": ["glue", "super-glue", "rainbow"],
                "transfer_method": ["multi-task"],
                "size": [
                    "00004",
                    "00010",
                    "00030",
                    "00091",
                    "00280",
                    "00865",
                    "02667",
                    "05334",
                    "08000",
                    "10667",
                    "13334",
                    "16000",
                ],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["single-task"],
        ),
        # the single-task learning curves table
        "single-task_learning-curves": (
            # path to root
            "learning-curves/t5",
            # experiment factors
            {
                "model_size": ["large"],
                "task": ["commonsenseqa", "joci"],
                "multiset": ["single-task"],
                "size": [
                    "00004",
                    "00010",
                    "00030",
                    "00091",
                    "00280",
                    "00865",
                    "02667",
                    "05334",
                    "08000",
                    "10667",
                    "13334",
                    "16000",
                ],
                "lr": ["lr-2.5e-4", "lr-1e-3", "lr-4e-3"],
                "split": ["validation_eval"],
            },
            # directories to ignore
            ["glue", "super-glue", "rainbow"],
        ),
    },
}
"""Table configurations for all of the experiments."""


# helper functions


def read_labels(fpath: str) -> List[str]:
    """Return the labels read from a ``fpath``.

    Parameters
    ----------
    fpath : str, required
        The path to the labels file. The file should have one label per
        line.

    Returns
    -------
    List[str]
        The labels located at ``fpath`` as a list of strings.
    """
    with open(fpath, "r") as f:
        labels = [ln.strip().lower() for ln in f]

    return labels


def parse_training_curve(dpath: str) -> List[Tuple[int, float]]:
    """Parse the training curve from ``dpath``.

    Parameters
    ----------
    dpath : str, required
        The path to the training curve's directory.

    Returns
    -------
    List[Tuple[int, float]]
        The training curve represented as a list of tuples of
        (step, score) number pairs.
    """
    # Parse and validate the directory structure.
    preds_fpattern = re.compile(
        r"(?P<task>[a-z]*(?:_\d+)?)_task_(?P<step>\d+)_predictions"
    )
    targets_fpattern = re.compile(r"(?P<task>[a-z]*(?:_\d+)?)_task_targets")

    tasks = set()
    preds_fpaths = set()
    targets_fpaths = set()

    for fpath in os.listdir(dpath):
        preds_fpattern_match = preds_fpattern.match(fpath)
        targets_fpattern_match = targets_fpattern.match(fpath)
        if preds_fpattern_match:
            preds_fpaths.add(fpath)
            tasks.add(preds_fpattern_match.groupdict()["task"])
        elif targets_fpattern_match:
            targets_fpaths.add(fpath)
            tasks.add(targets_fpattern_match.groupdict()["task"])
        else:
            raise IOError(
                f"The file path ({fpath}) did not match the"
                f" predictions or targets file path patterns."
            )

    if len(tasks) != 1:
        raise IOError(
            f"{dpath} should have files corresponding to exactly one task."
        )
    if len(preds_fpaths) != 10:
        raise IOError(
            f"There should be exactly 10 prediction files in {dpath}."
        )
    if len(targets_fpaths) != 1:
        raise IOError(f"There should be exactly 1 targets file in {dpath}.")

    # Pop the directory's task.

    task = tasks.pop()

    # Return the training curve as a list of (step, score) pairs.

    targets_fname = f"{task}_task_targets"
    training_curve = []
    for preds_fname in os.listdir(dpath):
        match = preds_fpattern.match(preds_fname)
        if match is None:
            continue

        step = int(match.groupdict()["step"])
        score = metrics.accuracy_score(
            y_pred=read_labels(os.path.join(dpath, preds_fname)),
            y_true=read_labels(os.path.join(dpath, targets_fname)),
        )

        training_curve.append((step, score))

    training_curve.sort()

    return training_curve


def parse_training_curves(
    dpath: str, ignore_dirs: Optional[List[str]] = None, **kwargs
):
    """Parse the training curves located at ``dpath``.

    Parse the training curves run with different factors
    (hyper-parameters) located at ``dpath`` into a nested dictionary
    structure.

    Parameters
    ----------
    dpath : str, required
        The path to the experiment's directory.
    ignore_dirs : Optional[List[str]], optional (default=None)
        An optional list of directory names to ignore when validating
        the directory structure. Defaults to ``None``.
    **kwargs, required
        Key-word arguments, each providing a list of strings for the
        directory names expected in deeper levels of the directory tree
        rooted at ``dpath``. Each directory name is assumed to be a
        factor for the experiment, and the directories are expected to
        be in order.

    Returns
    -------
    Dict
        Nested dictionaries mapping a sequence of factors to the
        corresponding training curve.
    """
    # Handle the base case.
    if len(kwargs) == 0:
        return parse_training_curve(dpath)

    # Pop the next set of factors.
    key = next(iter(kwargs))
    values = kwargs.pop(key)

    # Parse and validate the directory structure.
    found_subdirs = set(os.listdir(dpath)).difference(set(ignore_dirs or []))
    expected_subdirs = set(values)
    if found_subdirs != expected_subdirs:
        raise IOError(f"Directories in {dpath} did not match expected {key}.")

    # Return the mapping.
    return {
        value: parse_training_curves(
            dpath=os.path.join(dpath, value), ignore_dirs=ignore_dirs, **kwargs
        )
        for value in values
    }


def process_factor(key: str, value: str) -> Any:
    """Return ``value`` coerced to the proper type given ``key``.

    Parameters
    ----------
    key : str, required
        The name of the factor.
    value : str, required
        The factor's value.

    Returns
    -------
    Any
        ``value`` coerced to the appropriate type.
    """
    if key == "lr":
        return float(re.match(r"lr-(\d+(?:\.\d+)?e-?\d+)", value).groups()[0])
    if key == "split":
        if value.endswith("_eval"):
            value = value[: -len("_eval")]
        return str(value)
    if key == "size":
        return int(value)
    if key == "best_score":
        return float(value)
    if key == "training_curve":
        return list(value)

    return str(value)


# main function


@click.command()
@click.argument(
    "src", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "dst", type=click.Path(exists=False, dir_okay=True, file_okay=False)
)
def generate_tables(src: str, dst: str) -> None:
    """Generate tables from Rainbow's experimental results.

    Read the experimental results for Rainbow from SRC and write tables
    for the experiments out to DST, in the CSV format. Full curves from
    training are also written out in the JSON Lines format.
    """
    utils.configure_logging(clear=True)

    for experiment, tables_config in EXPERIMENT_TO_TABLES_CONFIG.items():
        for name, (path, factors, ignore_dirs) in tables_config.items():
            # Parse the data using the config.
            training_curves = parse_training_curves(
                dpath=os.path.join(src, experiment, path),
                ignore_dirs=ignore_dirs,
                **factors,
            )
            # Convert the training curve data from nested dictionaries
            # into two lists of dictionaries, using a depth-first
            # search. The first list of dictionaries, table, provides a
            # table grouping each hyper-parameter configuration to the
            # best score from training (i.e., early stopping). The
            # second list of dictionaries, training_curves_table, groups
            # each hyper-parameter configuration with the full training
            # curve.
            table = []
            training_curves_table = []
            values = []
            stack = [
                (value, children, 0)
                for value, children in list(training_curves.items())[::-1]
            ]
            while len(stack) > 0:
                # Pop the node off the stack.
                value, children, depth = stack.pop()
                # Truncate the values to the correct depth.
                values = values[:depth]
                # Update values with the current value.
                values.append(value)
                # Handle the node.
                if isinstance(children, list):
                    # The node is a leaf (training curves).
                    best_score = max(score for _, score in children)
                    table.append(values + [best_score])
                    training_curves_table.append(values + [children])
                elif isinstance(children, dict):
                    # The node is an internal node.
                    for value, children in list(children.items())[::-1]:
                        stack.append((value, children, depth + 1))
            # Write out the data.
            os.makedirs(os.path.join(dst, experiment, name))
            # Write the training curves.
            training_curves_table_path = os.path.join(
                dst, experiment, name, "training-curves.jsonl"
            )
            with open(training_curves_table_path, "w") as fout:
                fieldnames = list(factors.keys()) + ["training_curve"]
                for row in training_curves_table:
                    fout.write(
                        json.dumps(
                            {
                                factor: process_factor(factor, value)
                                for factor, value in zip(fieldnames, row)
                            }
                        )
                        + "\n"
                    )
            # Write the results table.
            table_path = os.path.join(dst, experiment, name, "table.csv")
            with open(table_path, "w") as fout:
                fieldnames = list(factors.keys()) + ["best_score"]
                writer = csv.DictWriter(
                    f=fout, fieldnames=fieldnames, dialect="unix"
                )

                writer.writeheader()
                for row in table:
                    writer.writerow(
                        {
                            factor: process_factor(factor, value)
                            for factor, value in zip(fieldnames, row)
                        }
                    )


if __name__ == "__main__":
    generate_tables()  # pylint: disable=no-value-for-parameter
