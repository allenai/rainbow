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
    # AlphaNLI
    "anli": Dataset(
        name="anli",
        url="gs://ai2-mosaic/public/alphanli/alphanli-train-dev.zip",
        checksum="24840b27553e93ec625ae020dbf78d92daeae4be31ebbd469a0c9f6f99ed1c8d",
        file_name="alphanli-train-dev.zip",
        feature_names=["obs1", "obs2", "hyp1", "hyp2"],
        splits={
            "train": Split(
                name="train",
                features_path="train.jsonl",
                labels_path="train-labels.lst",
                size=169654,
            ),
            "validation": Split(
                name="validation",
                features_path="dev.jsonl",
                labels_path="dev-labels.lst",
                size=1532,
            ),
        },
    ),
    # CosmosQA
    "cosmosqa": Dataset(
        name="cosmosqa",
        url="gs://ai2-mosaic/public/cosmosqa/cosmosqa-data.zip",
        checksum="d06bfef918240b34b6f86fdfd8215d15fea2abced0bfca8ab99004e3bce760ec",
        file_name="cosmosqa-data.zip",
        feature_names=[
            "context",
            "question",
            "answer0",
            "answer1",
            "answer2",
            "answer3",
        ],
        splits={
            "train": Split(
                name="train",
                features_path="train.jsonl",
                labels_path="train-labels.lst",
                size=25262,
            ),
            "validation": Split(
                name="validation",
                features_path="valid.jsonl",
                labels_path="valid-labels.lst",
                size=2985,
            ),
        },
    ),
    # HellaSWAG
    "hellaswag": Dataset(
        name="hellaswag",
        url="gs://ai2-mosaic/public/hellaswag/hellaswag-train-dev.zip",
        checksum="5d5d70300eff7af886c184477bb076fbfa24336cb300c52c3b6e62644d14d928",
        file_name="hellaswag-train-dev.zip",
        feature_names=["ctx", "ending_options"],
        splits={
            "train": Split(
                name="train",
                features_path="hellaswag-train-dev/train.jsonl",
                labels_path="hellaswag-train-dev/train-labels.lst",
                size=39905,
            ),
            "validation": Split(
                name="validation",
                features_path="hellaswag-train-dev/valid.jsonl",
                labels_path="hellaswag-train-dev/valid-labels.lst",
                size=10042,
            ),
        },
    ),
    # PhysicalIQA
    "physicaliqa": Dataset(
        name="physicaliqa",
        url="gs://ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
        checksum="54d32a04f59a7e354396f321723c8d7ec35cc6b08506563d8d1ffcc15ce98ddd",
        file_name="physicaliqa-train-dev.zip",
        feature_names=["goal", "sol1", "sol2"],
        splits={
            "train": Split(
                name="train",
                features_path="physicaliqa-train-dev/train.jsonl",
                labels_path="physicaliqa-train-dev/train-labels.lst",
                size=16113,
            ),
            "validation": Split(
                name="validation",
                features_path="physicaliqa-train-dev/dev.jsonl",
                labels_path="physicaliqa-train-dev/dev-labels.lst",
                size=1838,
            ),
        },
    ),
    # SocialIQA
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
    ),
    # WinoGrande
    "winogrande": Dataset(
        name="winogrande",
        url="gs://ai2-mosaic/public/winogrande/winogrande_1.1.zip",
        checksum="db997e35f11b014043531e7cd7ef30591022fd5946063e1e1e1416963d342fa5",
        file_name="winogrande_1.1.zip",
        feature_names=["sentence", "option1", "option2"],
        splits={
            "train": Split(
                name="train",
                features_path="winogrande_1.1/train_xl.jsonl",
                labels_path="winogrande_1.1/train_xl-labels.lst",
                size=40398,
            ),
            "validation": Split(
                name="validation",
                features_path="winogrande_1.1/dev.jsonl",
                labels_path="winogrande_1.1/dev-labels.lst",
                size=1267,
            ),
        },
    ),
}
"""The Rainbow Datasets."""
