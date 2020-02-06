"""Tests for rainbow.core."""

import unittest

import t5

from rainbow import core, datasets, tasks

# N.B. we must import tasks so that the tasks are registered and available for
# running tests.


class ProportionalRateTestCase(unittest.TestCase):
    """Tests for rainbow.core.proportional_rate."""

    def test_returns_training_set_size(self):
        for dataset in datasets.RAINBOW_DATASETS.values():
            task = t5.data.get_mixture_or_task(f"{dataset.name}_task")
            self.assertEqual(
                core.proportional_rate(task), dataset.splits["train"].size
            )


class EqualRateTestCase(unittest.TestCase):
    """Tests for rainbow.core.equal_rate."""

    def test_returns_one(self):
        for dataset in datasets.RAINBOW_DATASETS.values():
            task = t5.data.get_mixture_or_task(f"{dataset.name}_task")
            self.assertEqual(core.equal_rate(task), 1.0)
