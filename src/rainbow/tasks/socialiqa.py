"""The SocialIQA task."""

import json
from zipfile import ZipFile

from fairseq.tasks import register_task

from . import base


@register_task("socialiqa")
class SocialIQATask(base.MultipleChoiceTask):
    """The SocialIQA task."""

    _file_names = {
        "train": {
            "features": "socialiqa-train-dev/train.jsonl",
            "labels": "socialiqa-train-dev/train-labels.lst",
        },
        "valid": {
            "features": "socialiqa-train-dev/dev.jsonl",
            "labels": "socialiqa-train-dev/dev-labels.lst",
        },
    }

    def __init__(self, args, vocab):
        super().__init__(args, vocab)

        if args.num_classes != 3:
            raise ValueError("--num-classes must be equal to 3 on SocialIQA.")

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
                        [row["context"], row["question"], row["answerA"]],
                        [row["context"], row["question"], row["answerB"]],
                        [row["context"], row["question"], row["answerC"]],
                    ]
                    for i, choice in enumerate(choices):
                        features[i].append(choice)

            # preprocess the labels
            labels = []
            with dataset_zip.open(
                self._file_names[split]["labels"], "r"
            ) as labels_file:
                for ln in labels_file.readlines():
                    label = int(ln) - 1

                    labels.append(label)

        return features, labels
