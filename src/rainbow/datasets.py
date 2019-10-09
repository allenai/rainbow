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


class SocialIQADataset(Dataset):
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

    preprocessed_path_templates = {
        "atomic": "{split}.atomic-socialiqa.msg",
        "conceptnet": "{split}.conceptnet-socialiqa.msg",
        "original": "{split}.original-socialiqa.msg",
    }

    metric = metrics.accuracy_score

    @classmethod
    def _question_to_categories(cls, question) -> str:
        for start, end, categories in cls._QUESTION_TEMPLATES:
            if question.startswith(start) and question.endswith(end):
                return categories

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
        ids, features, labels = [], [], []

        split_path = os.path.join(
            self.data_dir,
            self.preprocessed_path_templates["atomic"].format(split=self.split),
        )
        with open(split_path, "rb") as split_file:
            for i, row in enumerate(msgpack.Unpacker(split_file, raw=False)):
                categories = self._question_to_categories(
                    row["features"]["question"]["text"]
                )

                if self.use_augmentation:
                    feature = (
                        row["features"]["context"]["text"],
                        row["features"]["context"][categories[0]],
                        row["features"]["context"][categories[1]],
                        row["features"]["question"]["text"],
                        [answer["text"] for answer in row["answers"]],
                    )
                else:
                    feature = (
                        row["features"]["context"]["text"],
                        row["features"]["question"]["text"],
                        [answer["text"] for answer in row["answers"]],
                    )

                ids.append(f"id{i}")
                features.append(feature)
                labels.append(row["label"])

        return ids, features, labels

    def __init__(
        self,
        data_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        use_augmentation: bool = False,
    ) -> None:
        super().__init__()

        splits = self._file_names.keys()
        if split not in splits:
            raise ValueError(
                f"Unrecognized value for split, split must be one of"
                f" {', '.join(splits)}."
            )

        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.use_augmentation = use_augmentation

        self.ids, self.features, self.labels = self._read_data()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, key: int) -> Tuple[str, Any, Any]:
        id_ = self.ids[key]
        feature = self.features[key]
        label = self.labels[key]

        if self.transform:
            feature = self.transform(feature)

        return id_, feature, label


DATASETS = {"socialiqa": SocialIQADataset}
