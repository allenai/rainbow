#! /usr/bin/env python

"""Generate latex tables from Rainbow's raw tables."""

import logging
import os
from typing import Dict, List, Optional

import click
import numpy as np
import pandas as pd
import re

from rainbow import utils


logger = logging.getLogger(__name__)


# constants

EXPERIMENT_TO_LATEX_TABLE_CONFIG = {
    "effect-of-size": {
        # the mixtures tables
        "mixtures": {
            "columns_to_drop": ["multiset", "rate", "split"],
            "column_renames": {"model_size": "model", "best_score": "accuracy"},
            "column_to_split_tables": None,
            "columns_to_aggregate": ["lr"],
            "columns_for_index": ["model"],
            "column_for_pivot": "task",
        },
        # the multiset learning curves tables
        "multiset_learning-curves": {
            "columns_to_drop": ["task", "multiset", "split"],
            "column_renames": {
                "model_size": "model",
                "transfer_method": "transfer",
                "best_score": "accuracy",
            },
            "column_to_split_tables": None,
            "columns_to_aggregate": ["lr"],
            "columns_for_index": ["model", "transfer"],
            "column_for_pivot": "size",
        },
        # the single-task learning curves
        "single-task_learning-curves": {
            "columns_to_drop": ["task", "multiset", "split"],
            "column_renames": {"model_size": "model", "best_score": "accuracy"},
            "column_to_split_tables": None,
            "columns_to_aggregate": ["lr"],
            "columns_for_index": ["model"],
            "column_for_pivot": "size",
        },
    },
    "transferring-multisets": {
        # the multiset full tasks tables
        "multiset_full-tasks": {
            "columns_to_drop": ["model_size", "split"],
            "column_renames": {
                "transfer_method": "transfer",
                "best_score": "accuracy",
            },
            "column_to_split_tables": None,
            "columns_to_aggregate": ["rate", "lr"],
            "columns_for_index": ["multiset", "transfer"],
            "column_for_pivot": "task",
        },
        # the multiset learning curves tables
        "multiset_learning-curves": {
            "columns_to_drop": ["model_size", "split"],
            "column_renames": {
                "transfer_method": "transfer",
                "best_score": "accuracy",
            },
            "column_to_split_tables": "task",
            "columns_to_aggregate": ["lr"],
            "columns_for_index": ["multiset", "transfer"],
            "column_for_pivot": "size",
        },
        # the single task full tasks tables
        "single-task_full-tasks": {
            "columns_to_drop": ["model_size", "multiset", "split"],
            "column_renames": {"best_score": "accuracy"},
            "column_to_split_tables": None,
            "columns_to_aggregate": ["lr"],
            "columns_for_index": [],
            "column_for_pivot": "task",
        },
        # the single task learning curves tables
        "single-task_learning-curves": {
            "columns_to_drop": ["model_size", "multiset", "split"],
            "column_renames": {"best_score": "accuracy"},
            "column_to_split_tables": None,
            "columns_to_aggregate": ["lr"],
            "columns_for_index": ["task"],
            "column_for_pivot": "size",
        },
    },
    "transferring-to-external-tasks": {
        # the multiset full tasks tables
        "multiset_full-tasks": {
            "columns_to_drop": ["model_size", "transfer_method", "split"],
            "column_renames": {"best_score": "accuracy"},
            "column_to_split_tables": None,
            "columns_to_aggregate": ["rate", "lr"],
            "columns_for_index": ["multiset"],
            "column_for_pivot": "task",
        },
        # the multiset learning curves tables
        "multiset_learning-curves": {
            "columns_to_drop": ["model_size", "transfer_method", "split"],
            "column_renames": {"best_score": "accuracy"},
            "column_to_split_tables": None,
            "columns_to_aggregate": ["lr"],
            "columns_for_index": ["task", "multiset"],
            "column_for_pivot": "size",
        },
        # the single task full tasks tables
        "single-task_full-tasks": {
            "columns_to_drop": ["model_size", "multiset", "split"],
            "column_renames": {"best_score": "accuracy"},
            "column_to_split_tables": None,
            "columns_to_aggregate": ["lr"],
            "columns_for_index": [],
            "column_for_pivot": "task",
        },
        # the single task learning curves tables
        "single-task_learning-curves": {
            "columns_to_drop": ["model_size", "multiset", "split"],
            "column_renames": {"best_score": "accuracy"},
            "column_to_split_tables": None,
            "columns_to_aggregate": ["lr"],
            "columns_for_index": ["task"],
            "column_for_pivot": "size",
        },
    },
    "transferring-knowledge-graphs": {
        # the multiset full tasks tables
        "multiset_full-tasks": {
            "columns_to_drop": ["model_size", "transfer_method", "split"],
            "column_renames": {
                "knowledge-graph": "knowledge",
                "best_score": "accuracy",
            },
            "column_to_split_tables": None,
            "columns_to_aggregate": ["rate", "lr"],
            "columns_for_index": ["multiset", "knowledge", "direction"],
            "column_for_pivot": "task",
        },
        # the multiset learning curves tables
        "multiset_learning-curves": {
            "columns_to_drop": ["model_size", "transfer_method", "split"],
            "column_renames": {
                "knowledge-graph": "knowledge",
                "best_score": "accuracy",
            },
            "column_to_split_tables": "task",
            "columns_to_aggregate": ["lr"],
            "columns_for_index": ["multiset", "knowledge"],
            "column_for_pivot": "size",
        },
        # the single task full tasks tables
        "single-task_full-tasks": {
            "columns_to_drop": ["model_size", "multiset", "split"],
            "column_renames": {"best_score": "accuracy"},
            "column_to_split_tables": None,
            "columns_to_aggregate": ["lr"],
            "columns_for_index": [],
            "column_for_pivot": "task",
        },
        # the single task learning curves tables
        "single-task_learning-curves": {
            "columns_to_drop": ["model_size", "multiset", "split"],
            "column_renames": {"best_score": "accuracy"},
            "column_to_split_tables": None,
            "columns_to_aggregate": ["lr"],
            "columns_for_index": ["task"],
            "column_for_pivot": "size",
        },
    },
}
"""Latex table configurations for all of the experiments."""


# helper functions


def write_latex_tables_for_config(
    src_fpath: str,
    dst_dpath: str,
    table_name: str,
    columns_to_drop: List[str],
    column_renames: Dict[str, str],
    column_to_split_tables: Optional[str],
    columns_to_aggregate: List[str],
    columns_for_index: List[str],
    column_for_pivot: str,
) -> None:
    """Write latex tables for the config arguments to disk.

    Parameters
    ----------
    src_fpath : str, required
        The path to the source table file.
    dst_dpath : str, required
        The path to the destination directory.
    table_name : str, required
        The name of the table.
    columns_to_drop : List[str], required
        Columns to drop from the raw table.
    column_renames : Dict[str], required
        A dictionary mapping column names to new names.
    column_to_split_tables : Optional[str], required
        The column whose values should be used for splitting the raw
        table into separate latex tables.
    columns_for_index : List[str], required
        The list of columns to use as the index for the table.
    columns_to_aggregate : List[str], required
        The columns to aggregate over by taking the max. These columns
        should be the (less interesting) hyper-parameters that were
        optimized over.
    column_for_pivot: str, required
        The column to pivot on for using as the table's final columns.

    Returns
    -------
    None.
    """
    # Read the data.
    df = pd.read_csv(src_fpath)

    # Drop columns.
    for col in columns_to_drop:
        del df[col]

    # Rename the columns.
    df = df.rename(columns=column_renames)

    # Transform the columns.
    if "task" in df:
        df["task"] = df["task"].apply(
            lambda x: {
                "anli": r"\anli{}",
                "hellaswag": r"\hellaswag{}",
                "cosmosqa": r"\cosmosqa{}",
                "physicaliqa": r"\physicaliqa{}",
                "socialiqa": r"\socialiqa{}",
                "winogrande": r"\winogrande{}",
                "commonsenseqa": r"\commonsenseqa{}",
                "joci": r"\joci{}",
            }[x]
        )
    if "multiset" in df:
        df["multiset"] = df["multiset"].apply(
            lambda x: {
                "rainbow": r"\rainbow{}",
                "glue": r"\glue{}",
                "super-glue": r"\superglue{}",
                "knowledge-graph": r"\none{}",
                "rainbow-knowledge-graph": r"\rainbow{}",
            }[x]
        )
    if "knowledge" in df:
        df["knowledge"] = df["knowledge"].apply(
            lambda x: {
                "atomic": r"\atomic{}",
                "conceptnet": r"\conceptnet{}",
                "comet": r"\both{}",
            }[x]
        )
    if "rate" in df:
        df["rate"] = df["rate"].apply(
            lambda x: {"equal": "equal", "proportional": "relative"}[x]
        )
    if "transfer" in df:
        df["transfer"] = df["transfer"].apply(
            lambda x: {
                "multi-task": "multitask",
                "multi-task-fine-tune": "fine-tune",
                "sequential-fine-tune": "sequential",
            }[x]
        )
    if "lr" in df:
        df["lr"] = df["lr"].apply(
            lambda x: np.format_float_scientific(x, precision=1, exp_digits=1)
        )
    df["accuracy"] *= 100

    # Group the data and write out the tables.
    subtables = (
        df[column_to_split_tables].unique()
        if column_to_split_tables is not None
        else ["all"]
    )
    for subtable in subtables:
        subdf = (
            df[df[column_to_split_tables] == subtable]
            if column_to_split_tables is not None
            else df
        )

        subdf = (
            subdf.groupby(
                [
                    col
                    for col in subdf.columns
                    if col not in columns_to_aggregate and col != "accuracy"
                ]
            )
            .max()
            .reset_index()
        )
        if len(columns_for_index) == 0:
            # Use a new, blank column as the index.
            subdf[""] = ""
            subdf = subdf.set_index([""])
        else:
            subdf = subdf.set_index(columns_for_index)
        subdf = subdf.pivot(index=subdf.index, columns=column_for_pivot)[
            "accuracy"
        ]
        subdf = subdf.rename(columns=lambda col: fr"\small{{{col}}}")

        table_str_lns = subdf.to_latex(
            buf=None,
            header=True,
            index=True,
            float_format="{:.1f}".format,
            sparsify=True,
            index_names=True,
            bold_rows=False,
            longtable=False,
            escape=False,
            multirow=True,
            multicolumn=True,
        ).split("\n")

        # Modify the table's header to add a multicolumn cell labeling the
        # column headers.
        table_str_lns[2:4] = [
            " &" * len(columns_for_index)
            + r"\multicolumn{"
            + str(len(subdf.columns))
            + r"}{c}{\small{"
            + column_for_pivot.capitalize()
            + r"}}\\ \cline{"
            + str(len(columns_for_index) + 1)
            + r"-"
            + str(len(columns_for_index) + len(subdf.columns))
            + r"}",
            re.match(
                r"^(" + r"[^&]*&" * len(columns_for_index) + r").*$",
                table_str_lns[3],
            ).groups()[0]
            + re.match(
                r"^" + r"[^&]*&" * len(columns_for_index) + r"(.*)$",
                table_str_lns[2],
            ).groups()[0],
        ]

        subtable_slug = re.sub(r"[\\{}]", "", str(subtable)).lower().strip()
        table_path = os.path.join(
            dst_dpath, f"{subtable_slug}.{table_name}.tex"
        )
        with open(table_path, "w") as fout:
            fout.write("\n".join(table_str_lns))


# main function


@click.command()
@click.argument(
    "src", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "dst", type=click.Path(exists=False, dir_okay=True, file_okay=False)
)
def generate_latex_tables(src: str, dst: str) -> None:
    """Generate latex tables from Rainbow's raw tables.

    Read the raw tables from SRC and write out latex tables to DST. The
    generated tables assume the document has macros for \anli{},
    \cosmosqa{}, \hellaswag{}, \physicaliqa{}, \socialiqa{},
    \winogrande{}, \commonsenseqa{}, \joci{}, \rainbow{}, \glue{},
    \superglue{}, \none{}, \atomic{}, \conceptnet{}, and \both{}.
    """
    utils.configure_logging(clear=True)

    for experiment, tables_config in EXPERIMENT_TO_LATEX_TABLE_CONFIG.items():
        for name, config in tables_config.items():
            src_fpath = os.path.join(src, experiment, name, "table.csv")
            dst_dpath = os.path.join(dst, experiment, name)
            os.makedirs(dst_dpath)
            write_latex_tables_for_config(
                src_fpath=src_fpath,
                dst_dpath=dst_dpath,
                table_name=name,
                **config,
            )


if __name__ == "__main__":
    generate_latex_tables()  # pylint: disable=no-value-for-parameter
