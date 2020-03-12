"""Tests for rainbow.core."""

import csv
import os
import tempfile
import unittest

import t5
import tensorflow as tf

from rainbow import core, preprocessors
from .. import settings, utils


class CsvTaskTestCase(unittest.TestCase):
    """Tests for rainbow.core.CsvTask."""

    def setUp(self):
        # Copy the CsvTask fixtures to disk.
        self.temp_dir = tempfile.TemporaryDirectory()

        for split, path in settings.CSV_TASK_SPLIT_FIXTURES.items():
            utils.copy_pkg_resource_to_disk(
                pkg="tests",
                src=path,
                dst=os.path.join(self.temp_dir.name, os.path.basename(path)),
            )

        self.split_to_filepattern = {
            split: os.path.join(self.temp_dir.name, os.path.basename(path))
            for split, path in settings.CSV_TASK_SPLIT_FIXTURES.items()
        }

    def tearDown(self):
        # Clean up the CsvTask fixtures that were copied to disk.
        self.temp_dir.cleanup()

    def make_test_task(self, truncate_to):
        # A helper method for creating tests
        return core.CsvTask(
            name="test_task",
            # dataset configuration and location
            split_to_filepattern=self.split_to_filepattern,
            num_input_examples=settings.CSV_TASK_SPLIT_NUM_EXAMPLES,
            text_preprocessor=[
                preprocessors.make_add_field_names_preprocessor(
                    field_indices=[1, 2], field_names=["inputs", "targets"]
                )
            ],
            sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
            metric_fns=[t5.evaluation.metrics.accuracy],
            # csv parsing
            record_defaults=[tf.int32, tf.string, tf.string],
            compression_type=None,
            buffer_size=None,
            header=True,
            field_delim=",",
            use_quote_delim=True,
            na_value="",
            select_cols=None,
            # dataset truncation
            truncate_to=truncate_to,
            # args for the task class
            postprocess_fn=t5.data.postprocessors.lower_text,
        )

    def test_creates_datasets_correctly(self):
        task = self.make_test_task(truncate_to=None)

        # test train
        train = task.get_dataset(
            sequence_length={"inputs": 512, "targets": 512},
            split="train",
            use_cached=False,
            shuffle=False,
        )
        with open(self.split_to_filepattern["train"], "r") as train_file:
            train_csv = csv.DictReader(train_file)
            for actual, expected in zip(train, train_csv):
                self.assertEqual(
                    actual["inputs_plaintext"].numpy().decode(),
                    expected["inputs"],
                )
                self.assertEqual(
                    actual["targets_plaintext"].numpy().decode(),
                    expected["targets"],
                )

        # test validation
        validation = task.get_dataset(
            sequence_length={"inputs": 512, "targets": 512},
            split="validation",
            use_cached=False,
            shuffle=False,
        )
        with open(
            self.split_to_filepattern["validation"], "r"
        ) as validation_file:
            validation_csv = csv.DictReader(validation_file)
            for actual, expected in zip(validation, validation_csv):
                self.assertEqual(
                    actual["inputs_plaintext"].numpy().decode(),
                    expected["inputs"],
                )
                self.assertEqual(
                    actual["targets_plaintext"].numpy().decode(),
                    expected["targets"],
                )

    def test_it_truncates_the_training_set_properly(self):
        # test that there's no truncation when truncate_to is None
        task_full = self.make_test_task(truncate_to=None)
        train_full = task_full.get_dataset(
            sequence_length={"inputs": 512, "targets": 512},
            split="train",
            use_cached=False,
            shuffle=False,
        )
        len_train_full = sum(1 for _ in train_full)
        self.assertEqual(len_train_full, 5)

        # test that it truncates when truncate_to is an int
        task_truncated = self.make_test_task(truncate_to=3)
        train_truncated = task_truncated.get_dataset(
            sequence_length={"inputs": 512, "targets": 512},
            split="train",
            use_cached=False,
            shuffle=False,
        )
        len_train_truncated = sum(1 for _ in train_truncated)
        self.assertEqual(len_train_truncated, 3)

    def test_it_does_not_truncate_validation(self):
        # test that it doesn't truncate validation when truncate_to is
        # None
        task_full = self.make_test_task(truncate_to=None)
        validation_full = task_full.get_dataset(
            sequence_length={"inputs": 512, "targets": 512},
            split="validation",
            use_cached=False,
            shuffle=False,
        )
        len_validation_full = sum(1 for _ in validation_full)
        self.assertEqual(len_validation_full, 5)

        # test that it doesn't truncate validation when truncate_to is
        # an int
        task_truncated = self.make_test_task(truncate_to=3)
        validation_truncated = task_truncated.get_dataset(
            sequence_length={"inputs": 512, "targets": 512},
            split="validation",
            use_cached=False,
            shuffle=False,
        )
        len_validation_truncated = sum(1 for _ in validation_truncated)
        self.assertEqual(len_validation_truncated, 5)

    def test_that_truncation_is_deterministic(self):
        task = self.make_test_task(truncate_to=3)
        train = task.get_dataset(
            sequence_length={"inputs": 512, "targets": 512},
            split="train",
            use_cached=False,
            shuffle=False,
        )
        expected_train = [
            {"inputs": "foo 0", "targets": "0"},
            {"inputs": "bar 1", "targets": "1"},
            {"inputs": "bar 0", "targets": "0"},
        ]
        for actual, expected in zip(train, expected_train):
            self.assertEqual(
                actual["inputs_plaintext"].numpy().decode(), expected["inputs"]
            )
            self.assertEqual(
                actual["targets_plaintext"].numpy().decode(),
                expected["targets"],
            )
