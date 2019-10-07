"""Test rainbow.utils."""

import unittest

from rainbow import utils


class TransposeDictionaryTestCase(unittest.TestCase):
    """Test rainbow.utils.transpose_dictionary."""

    def test_empty_dictionary_returns_empty_list(self):
        self.assertEqual(utils.transpose_dictionary({}), [])

    def test_transposes_dictionary(self):
        self.assertEqual(
            utils.transpose_dictionary({"a": [1], "b": [2]}), [{"a": 1, "b": 2}]
        )
        self.assertEqual(
            utils.transpose_dictionary({"a": ["A", "X"], "b": ["B", "Y"]}),
            [{"a": "A", "b": "B"}, {"a": "X", "b": "Y"}],
        )
