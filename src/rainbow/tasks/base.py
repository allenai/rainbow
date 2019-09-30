"""Base classes for various types of tasks."""

from fairseq.data import encoders, Dictionary
from fairseq.models import build_model
from fairseq.tasks import FairseqTask


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
