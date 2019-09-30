"""The CosmosQA task."""

import json
from zipfile import ZipFile

from fairseq.tasks import register_task

from . import base


@register_task("cosmosqa")
class CosmosQATask(base.MultipleChoiceTask):
    """The CosmosQA task."""

    _file_names = {
        "train": {"features": "train.jsonl", "labels": "train-labels.lst"},
        "valid": {"features": "valid.jsonl", "labels": "valid-labels.lst"},
    }

    def __init__(self, args, vocab):
        super().__init__(args, vocab)

        if args.num_classes != 4:
            raise ValueError("--num-classes must be equal to 4 on CosmosQA.")

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
                        [row["context"], row["question"], row["answer0"]],
                        [row["context"], row["question"], row["answer1"]],
                        [row["context"], row["question"], row["answer2"]],
                        [row["context"], row["question"], row["answer3"]],
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
