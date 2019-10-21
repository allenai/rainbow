"""Datasets."""

import json
import os
from typing import Any, Callable, List, Optional, Tuple
from zipfile import ZipFile

import msgpack
from sklearn import metrics
from torch.utils.data import Dataset

from . import settings
from .features import TextFeature
from .instances import MultipleChoiceInstance


AUGMENTATION_TYPES = ["original", "atomic_text", "atomic_vector"]
"""The types of feature augmentation available."""


ATOMIC_RELATIONS = [
    "o_effect",
    "o_react",
    "o_want",
    "x_attr",
    "x_effect",
    "x_intent",
    "x_need",
    "x_react",
    "x_want",
]
"""The ATOMIC relations."""


class MultipleChoiceDataset(Dataset):
    """A base class for multiple choice datasets."""

    preprocessed_path_templates = {
        "atomic": "{split}.atomic-{name}.msg",
        "conceptnet": "{split}.conceptnet-{name}.msg",
        "original": "{split}.original-{name}.msg",
    }

    name = None
    metric = None
    splits = None

    @classmethod
    def read_raw_instances(
        cls, dataset_path, split
    ) -> List[MultipleChoiceInstance]:
        raise NotImplementedError

    def _read_data(self):
        raise NotImplementedError

    def __init__(
        self,
        data_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        transform_embedding: Optional[Callable] = None,
        augmentation_type: str = "original",
    ) -> None:
        super().__init__()

        if split not in self.splits:
            raise ValueError(
                f"Unrecognized value for split, split must be one of"
                f" {', '.join(self.splits)}."
            )

        if augmentation_type not in AUGMENTATION_TYPES:
            raise ValueError(
                f"Unrecognized augmentation_type ({augmentation_type})."
            )

        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.transform_embedding = transform_embedding
        self.augmentation_type = augmentation_type

        self.ids, self.features, self.embeddings, self.labels = (
            self._read_data()
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, key: int) -> Tuple[str, Any, Any]:
        id_ = self.ids[key]
        feature = self.features[key]
        embedding = self.embeddings[key]
        label = self.labels[key]

        if self.transform:
            feature, label = self.transform(feature, label)

        if self.transform_embedding:
            embedding = self.transform_embedding(embedding)

        return id_, feature, embedding, label


class SocialIQADataset(MultipleChoiceDataset):
    """The SocialIQA dataset."""

    _file_names = {
        "train": {
            "features": "socialiqa-train-dev/train.jsonl",
            "labels": "socialiqa-train-dev/train-labels.lst",
        },
        "dev": {
            "features": "socialiqa-train-dev/dev.jsonl",
            "labels": "socialiqa-train-dev/dev-labels.lst",
        },
    }

    _QUESTION_TEMPLATES = [
        ("How would", "feel as a result?", ["x_react", "o_react"]),
        ("What will happen to", "?", ["x_effect", "o_effect"]),
        ("How would", "feel afterwards?", ["x_react", "o_react"]),
        ("What does", "need to do before this?", ["x_intent", "x_need"]),
        ("Why did", "do this?", ["x_intent", "x_need"]),
        ("How would you describe", "?", ["x_attr", "x_effect"]),
        ("What will", "want to do next?", ["x_want", "o_want"]),
    ]

    name = "socialiqa"
    splits = ["train", "dev"]
    metric = metrics.accuracy_score

    @classmethod
    def _question_to_relations(cls, question) -> str:
        for start, end, relations in cls._QUESTION_TEMPLATES:
            if question.startswith(start) and question.endswith(end):
                return relations

        return ["x_want", "o_want"]

    @classmethod
    def read_raw_instances(
        cls, dataset_path, split
    ) -> List[MultipleChoiceInstance]:
        with ZipFile(dataset_path) as dataset_zip:
            features_path = cls._file_names[split]["features"]
            with dataset_zip.open(features_path, "r") as features_file:
                features = [json.loads(ln) for ln in features_file]

            labels_path = cls._file_names[split]["labels"]
            with dataset_zip.open(labels_path, "r") as labels_file:
                labels = [ln.decode().strip() for ln in labels_file]

        return [
            MultipleChoiceInstance(
                features={
                    "context": TextFeature(text=feature["context"]),
                    "question": TextFeature(text=feature["question"]),
                },
                answers=[
                    TextFeature(text=feature["answerA"]),
                    TextFeature(text=feature["answerB"]),
                    TextFeature(text=feature["answerC"]),
                ],
                label=int(label) - 1,
            )
            for feature, label in zip(features, labels)
        ]

    def _read_data(self):
        ids, features, embeddings, labels = [], [], [], []

        split_path = os.path.join(
            self.data_dir,
            self.preprocessed_path_templates["atomic"].format(
                split=self.split, name=self.name
            ),
        )
        with open(split_path, "rb") as split_file:
            for i, row in enumerate(msgpack.Unpacker(split_file, raw=False)):
                relations = self._question_to_relations(
                    row["features"]["question"]["text"]
                )

                if self.augmentation_type == "original":
                    feature = (
                        row["features"]["context"]["text"],
                        row["features"]["question"]["text"],
                        [answer["text"] for answer in row["answers"]],
                    )
                    embedding = ()
                elif self.augmentation_type == "atomic_text":
                    feature = (
                        row["features"]["context"]["text"],
                        row["features"]["context"][relations[0]],
                        row["features"]["context"][relations[1]],
                        row["features"]["question"]["text"],
                        [answer["text"] for answer in row["answers"]],
                    )
                    embedding = ()
                elif self.augmentation_type == "atomic_vector":
                    feature = (
                        row["features"]["context"]["text"],
                        row["features"]["question"]["text"],
                        [answer["text"] for answer in row["answers"]],
                    )
                    embedding = tuple(
                        row["features"]["context"][f"{relation}_embeddings"]
                        for relation in ATOMIC_RELATIONS
                    )

                ids.append(f"id{i}")
                features.append(feature)
                embeddings.append(embedding)
                labels.append(row["label"])

        return ids, features, embeddings, labels


DATASETS = {SocialIQADataset.name: SocialIQADataset}
