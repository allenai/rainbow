"""Dataset definitions for rainbow."""

from typing import Dict, List

import attr


# core classes


@attr.s(auto_attribs=True, frozen=True, kw_only=True, slots=True)
class Split:
    """A single split (train, validation, test, ...) of a dataset."""

    name: str
    features_path: str
    labels_path: str
    size: int


@attr.s(auto_attribs=True, frozen=True, kw_only=True, slots=True)
class Dataset:
    """A class representing a rainbow dataset."""

    name: str
    url: str
    checksum: str
    file_name: str
    feature_names: List[str]
    splits: Dict[str, Split]


# constants

RAINBOW_DATASETS = {
    "socialiqa": Dataset(
        name="socialiqa",
        url="gs://ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip",
        checksum="ee073914a0fc33265cfbcfc50ec20df9b2e07809c3f420f599138d1f394ef5c3",
        file_name="socialiqa-train-dev.zip",
        feature_names=["context", "question", "answerA", "answerB", "answerC"],
        splits={
            "train": Split(
                name="train",
                features_path="socialiqa-train-dev/train.jsonl",
                labels_path="socialiqa-train-dev/train-labels.lst",
                size=33410,
            ),
            "validation": Split(
                name="validation",
                features_path="socialiqa-train-dev/dev.jsonl",
                labels_path="socialiqa-train-dev/dev-labels.lst",
                size=1954,
            ),
        },
    )
}
"""The Rainbow Datasets."""
