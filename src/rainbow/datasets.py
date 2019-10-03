"""Datasets."""

import json
from zipfile import ZipFile

from .features import TextFeature
from .instances import MultipleChoiceInstance


class Dataset:
    """An abstract base class for datasets."""

    pass


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

    @classmethod
    def _read_instances(cls, dataset_path, split):
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
                label=label,
            )
            for feature, label in zip(features, labels)
        ]

    def __init__(self, dataset_path, split):
        splits = self._file_names.keys()
        if split not in splits:
            raise ValueError(
                f"Unrecognized value for split, split must be one of"
                f" {', '.join(splits)}."
            )

        # bind arguments to the instance
        self.dataset_path = dataset_path
        self.split = split

        # load the datset's instances
        self.instances = self._read_instances(dataset_path, split)
