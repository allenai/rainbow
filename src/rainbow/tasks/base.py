"""Base classes for various types of tasks."""

from fairseq.data import (
    encoders,
    Dictionary,
    IdDataset,
    ListDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
)
from fairseq.models import build_model
from fairseq.tasks import FairseqTask
import numpy as np
import torch


# base tasks


# The following class was adapted from
# https://github.com/pytorch/fairseq/blob/master/examples/roberta/commonsense_qa/commonsense_qa_task.py
# at commit e073ddfe46d71f80340c2600a7bf9aed2696c692. See the license below.
#
# MIT License
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class MultipleChoiceTask(FairseqTask):
    """A base class for multiple choice tasks."""

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "dataset_path", type=str, help="The path to the dataset."
        )
        parser.add_argument(
            "vocab",
            type=str,
            help="The path to your vocabulary (dict.txt) file.",
        )
        parser.add_argument("--num-classes", type=int)

    @classmethod
    def load_dictionary(cls, filename):
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        if args.criterion != "sentence_ranking":
            raise ValueError(
                f"The criterion for multiple choice must be sentence ranking."
            )

        # load data and label dictionaries
        vocab = cls.load_dictionary(args.vocab)
        print("| dictionary: {} types".format(len(vocab)))

        return cls(args, vocab)

    def __init__(self, args, vocab):
        super().__init__(args)
        self.bpe = encoders.build_bpe(args)
        self.vocab = vocab
        self.mask = vocab.add_symbol("<mask>")

    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        return self.vocab

    def build_model(self, args):
        model = build_model(args, self)

        model.register_classification_head(
            "sentence_classification_head", num_classes=1
        )

        return model

    def load_features_and_labels(self, split):
        raise NotImplementedError

    def load_dataset(self, split, combine=False, **kwargs):
        features, labels = self.load_features_and_labels(split=split)

        # apply BPE and index the tokens
        input_tokens = [
            [
                torch.cat(  # pylint: disable=no-member
                    [
                        self.vocab.encode_line(
                            self.bpe.encode(x),
                            append_eos=True,
                            add_if_not_exist=False,
                        ).long()
                        for x in choice
                    ]
                )
                for choice in choices
            ]
            for choices in features
        ]
        input_lengths = [
            np.array([len(choice) for choice in choices])
            for choices in input_tokens
        ]
        # Convert the indexed tokens into datasets
        input_tokens = [
            ListDataset(x, y) for x, y in zip(input_tokens, input_lengths)
        ]
        input_lengths = [ListDataset(y) for y in input_lengths]

        # create the full dataset
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
