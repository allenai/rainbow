"""Dataset definitions for rainbow."""

from typing import Dict, List

from . import utils


# N.B. The names and sizes for the datasets must correspond to the names
# and sizes used during dataset preparation (see
# $REPO/bin/prepare.py and $REPO/src/rainbow/preparation/).


# core classes


@utils.data
class Split:
    """A single split of a dataset."""

    name: str
    size: int


@utils.data
class Dataset:
    """A dataset."""

    name: str
    splits: Dict[str, Split]


# constants

RAINBOW_DATASETS = {
    # AlphaNLI
    "anli": Dataset(
        name="anli",
        splits={
            "train": Split(name="train", size=169654),
            "validation": Split(name="validation", size=1532),
        },
    ),
    # CosmosQA
    "cosmosqa": Dataset(
        name="cosmosqa",
        splits={
            "train": Split(name="train", size=25262),
            "validation": Split(name="validation", size=2985),
        },
    ),
    # HellaSWAG
    "hellaswag": Dataset(
        name="hellaswag",
        splits={
            "train": Split(name="train", size=39905),
            "validation": Split(name="validation", size=10042),
        },
    ),
    # PhysicalIQA
    "physicaliqa": Dataset(
        name="physicaliqa",
        splits={
            "train": Split(name="train", size=16113),
            "validation": Split(name="validation", size=1838),
        },
    ),
    # SocialIQA
    "socialiqa": Dataset(
        name="socialiqa",
        splits={
            "train": Split(name="train", size=33410),
            "validation": Split(name="validation", size=1954),
        },
    ),
    # WinoGrande
    "winogrande": Dataset(
        name="winogrande",
        splits={
            "train": Split(name="train", size=40398),
            "validation": Split(name="validation", size=1267),
        },
    ),
}
"""Rainbow datasets."""


KNOWLEDGE_GRAPH_DATASETS = {
    # ATOMIC
    "atomic": Dataset(
        name="atomic",
        splits={
            "train": Split(name="train", size=709996),
            "validation": Split(name="validation", size=79600),
        },
    ),
    # ConceptNet
    "conceptnet": Dataset(
        name="conceptnet",
        splits={
            "train": Split(name="train", size=100000),
            "validation": Split(name="validation", size=1200),
        },
    ),
}
"""Commonsense knowledge graph datasets."""
