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

    def load_dataset(self, split, combine=False, **kwargs):
        with ZipFile(self.args.dataset_path) as dataset_zip:
            # preprocess features
            input_tokens = [[] for _ in range(self.args.num_classes)]
            input_lengths = [[] for _ in range(self.args.num_classes)]
            with dataset_zip.open(
                self._file_names[split]["features"], "r"
            ) as features_file:
                for ln in features_file.readlines():
                    features = json.loads(ln)

                    choices = [
                        [features["obs1"], features["hyp1"], features["obs2"]],
                        [features["obs1"], features["hyp2"], features["obs2"]],
                    ]
                    for i, choice in enumerate(choices):
                        if self.bpe:
                            # apply BPE if its available
                            choice = [self.bpe.encode(x) for x in choice]
                        # index the tokens with vocabulary
                        choice = torch.cat(  # pylint: disable=no-member
                            [
                                self.vocab.encode_line(
                                    x, append_eos=True, add_if_not_exist=False
                                ).long()
                                for x in choice
                            ]
                        )
                        input_tokens[i].append(choice)
                        input_lengths[i].append(len(choice))

            # preprocess the labels
            labels = []
            with dataset_zip.open(
                self._file_names[split]["labels"], "r"
            ) as labels_file:
                for ln in labels_file.readlines():
                    label = int(ln) - 1

                    labels.append(label)

        input_lengths = [np.array(x) for x in input_lengths]
        input_tokens = [
            ListDataset(x, y) for x, y in zip(input_tokens, input_lengths)
        ]
        input_lengths = [ListDataset(y) for y in input_lengths]

        # create the dataset
        dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(input_tokens[0], reduce=True),
                "target": RawLabelDataset(labels),
                **{
                    f"net_input{i+1}": {
                        "src_tokens": RightPadDataset(
                            input_tokens[i],
                            pad_idx=self.source_dictionary.pad(),
                        ),
                        "src_lengths": input_lengths[i],
                    }
                    for i in range(self.args.num_classes)
                },
            },
            sizes=np.maximum.reduce(  # pylint: disable=no-member
                [x.sizes for x in input_tokens]
            ),
        )

        # shuffle the dataset
        dataset = SortDataset(
            dataset, sort_order=[np.random.permutation(len(dataset))]
        )

        self.datasets[split] = dataset

        return dataset
