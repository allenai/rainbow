"""The PhysicalIQA task."""

import json
from zipfile import ZipFile

from fairseq.tasks import register_task

from . import base


@register_task("physicaliqa")
class PhysicalIQATask(base.MultipleChoiceTask):
    """The PhysicalIQA task."""

    _file_names = {
        "train": {
            "features": "physicaliqa-train-dev/train.jsonl",
            "labels": "physicaliqa-train-dev/train-labels.lst",
        },
        "valid": {
            "features": "physicaliqa-train-dev/dev.jsonl",
            "labels": "physicaliqa-train-dev/dev-labels.lst",
        },
    }

    def __init__(self, args, vocab):
        super().__init__(args, vocab)

        if args.num_classes != 2:
            raise ValueError("--num-classes must be equal to 2 on PhysicalIQA.")

    def load_features_and_labels(self, split):
        with ZipFile(self.args.dataset_path) as dataset_zip:
            # preprocess the features
            features = [[] for _ in range(self.args.num_classes)]
            with dataset_zip.open(
                self._file_names[split]["features"], "r"
            ) as features_file:
                for ln in features_file.readlines():
                    row = json.loads(ln)

                    choices = [
                        [row["goal"], row["sol1"]],
                        [row["goal"], row["sol2"]],
                    ]
                    for i, choice in enumerate(choices):
                        features[i].append(choice)

            # preprocess the labels
            labels = []
            with dataset_zip.open(
                self._file_names[split]["labels"], "r"
            ) as labels_file:
                for ln in labels_file.readlines():
                    label = int(ln)

                    labels.append(label)

        return features, labels
