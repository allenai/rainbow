"""The AlphaNLI task."""

import json
from zipfile import ZipFile

from fairseq.data import (
    IdDataset,
    ListDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
)
from fairseq.tasks import register_task
import numpy as np
import torch

from . import base


@register_task("anli")
class AlphaNLITask(base.MultipleChoiceTask):
    """The AlphaNLI task."""

    _file_names = {
        "train": {"features": "train.jsonl", "labels": "train-labels.lst"},
        "valid": {"features": "dev.jsonl", "labels": "dev-labels.lst"},
    }

    def __init__(self, args, vocab):
        super().__init__(args, vocab)

        if args.num_classes != 2:
            raise ValueError("--num-classes must be equal to 2 on AlphaNLI.")

    def load_features_and_labels(self, split):
        with ZipFile(self.args.dataset_path) as dataset_zip:
            # preprocess features
            features = [[] for _ in range(self.args.num_classes)]
            with dataset_zip.open(
                self._file_names[split]["features"], "r"
            ) as features_file:
                for ln in features_file.readlines():
                    row = json.loads(ln)

                    choices = [
                        [row["obs1"], row["hyp1"], row["obs2"]],
                        [row["obs1"], row["hyp2"], row["obs2"]],
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
