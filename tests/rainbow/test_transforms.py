"""Tests for rainbow.transforms."""

import unittest
from unittest.mock import Mock

import pytest
from transformers import RobertaTokenizer

from rainbow import transforms


class MapTestCase(unittest.TestCase):
    """Test rainbow.transforms.Map."""

    def test_mapping_the_empty_list(self):
        transform = transforms.Map(lambda x: x ** 2)
        transform.fit([])

        self.assertEqual(transform([]), [])

    def test_it_maps_a_mocked_function(self):
        mock = Mock(return_value="foo")

        transform = transforms.Map(mock)
        transform.fit([])

        self.assertEqual(transform([1, 2, 3]), ["foo", "foo", "foo"])

        mock.assert_any_call(1)
        mock.assert_any_call(2)
        mock.assert_any_call(3)

        self.assertEqual(mock.call_count, 3)

    def test_it_maps_an_actual_function(self):
        transform = transforms.Map(transform=lambda x: x ** 2)
        transform.fit([])

        self.assertEqual(transform([1, 2, 3]), [1, 4, 9])

        transform = transforms.Map(transform=str.lower)
        transform.fit([])

        self.assertEqual(transform(["Aa", "BB", "cC"]), ["aa", "bb", "cc"])

    def test_mapping_different_sequence_types(self):
        transform = transforms.Map(transform=lambda x: x ** 2)
        transform.fit([])

        self.assertEqual(transform((1, 2, 3)), [1, 4, 9])


class ComposeTestCase(unittest.TestCase):
    """Test rainbow.transforms.Compose."""

    def test_fit(self):
        # test fit does not throw an error if one of the transforms doesn't
        # have a fit method
        transform = transforms.Compose([lambda x: x ** 2])
        transform.fit([1, 2, 3])

        # test fit fits the transforms
        mock1 = Mock(return_value="bar")
        mock2 = Mock(return_value="baz")

        transform = transforms.Compose([mock1, mock2])
        transform.fit([1, 2, 3])

        mock1.fit.assert_called_with([1, 2, 3])
        mock2.fit.assert_called_with(["bar", "bar", "bar"])

    def test___call__(self):
        transform = transforms.Compose([lambda x: x + 1])

        # test __call__ throws an error if transform hasn't been fitted
        with self.assertRaises(ValueError):
            transform(1)

        transform.fit([])

        self.assertEqual(transform(1), 2)

    def test_empty_list_is_identity(self):
        transform = transforms.Compose([])
        transform.fit([])

        self.assertEqual(transform(1), 1)
        self.assertEqual(transform("a"), "a")

    def test_composes_mocked_functions(self):
        mock1 = Mock(return_value="bar")
        mock2 = Mock(return_value="baz")

        transform = transforms.Compose([mock1, mock2])
        transform.fit([])

        self.assertEqual(transform("foo"), "baz")

        mock1.assert_called_with("foo")
        mock2.assert_called_with("bar")

    def test_composes_actual_functions(self):
        transform = transforms.Compose([lambda x: x ** 2, lambda x: x + 1])
        transform.fit([])

        self.assertEqual(transform(-2), 5)

        transform = transforms.Compose([lambda s: s.lower(), lambda s: s + "b"])
        transform.fit([])

        self.assertEqual(transform("A"), "ab")


class DistributeContextTransformTestCase(unittest.TestCase):
    """Test rainbow.transforms.DistributeContextTransform."""

    def test_fit(self):
        transform = transforms.DistributeContextTransform()
        transform.fit([])

    def test___call__(self):
        transform = transforms.DistributeContextTransform()

        # test __call__ throws an error if transform hasn't been fitted
        with self.assertRaises(ValueError):
            transform(["a"])

        transform.fit([])

        self.assertEqual(
            transform(("The question", ("A", "B", "C"))),
            [
                ["The question", "A"],
                ["The question", "B"],
                ["The question", "C"],
            ],
        )
        self.assertEqual(
            transform(
                (
                    "Some context about the question",
                    "The question",
                    ("A", "B", "C"),
                )
            ),
            [
                ["Some context about the question", "The question", "A"],
                ["Some context about the question", "The question", "B"],
                ["Some context about the question", "The question", "C"],
            ],
        )


class LinearizeTransformTestCase(unittest.TestCase):
    """Test rainbow.transforms.LinearizeTransform."""

    @pytest.mark.slow
    def test_fit(self):
        # check the empty list
        transform = transforms.LinearizeTransform(
            tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            max_sequence_length=10,
            truncation_strategy="beginning",
        )
        transform.fit([])
        self.assertEqual(transform.max_sequence_length, 0)

        # check that the size goes down on a non-empty list
        transform = transforms.LinearizeTransform(
            tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            max_sequence_length=10,
            truncation_strategy="beginning",
        )
        transform.fit([["I"], ["oh"], ["I am"]])
        self.assertEqual(transform.max_sequence_length, 4)

        # check that the size does not go up
        transform = transforms.LinearizeTransform(
            tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            max_sequence_length=3,
            truncation_strategy="beginning",
        )
        transform.fit([["I am."]])
        self.assertEqual(transform.max_sequence_length, 3)

    @pytest.mark.slow
    def test___call__(self):
        transform = transforms.LinearizeTransform(
            tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            max_sequence_length=10,
            truncation_strategy="beginning",
        )

        # test __call__ throws an error if transform hasn't been fitted
        with self.assertRaises(ValueError):
            transform(["a"])

    @pytest.mark.slow
    def test_it_transforms_text_correctly(self):
        # test the input sequence has length 1
        transform = transforms.LinearizeTransform(
            tokenizer=RobertaTokenizer.from_pretrained(
                "roberta-base", do_lower_case=True
            ),
            max_sequence_length=10,
            truncation_strategy="beginning",
        )
        # artificially fit the transform
        transform._fitted = True

        transformed = transform(["This sentence is for testing."])
        self.assertEqual(
            transformed["input_ids"], [0, 713, 3645, 16, 13, 3044, 4, 2, 0, 0]
        )
        self.assertEqual(
            transformed["input_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        )

        # test the input sequence has length 2
        transform = transforms.LinearizeTransform(
            tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            max_sequence_length=18,
            truncation_strategy="beginning",
        )
        # artificially fit the transform
        transform._fitted = True

        transformed = transform(
            [
                "This sentence is for testing.",
                "This sentence is also for testing.",
            ]
        )
        self.assertEqual(
            transformed["input_ids"],
            [
                0,
                713,
                3645,
                16,
                13,
                3044,
                4,
                2,
                2,
                713,
                3645,
                16,
                67,
                13,
                3044,
                4,
                2,
                0,
            ],
        )
        self.assertEqual(
            transformed["input_mask"],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        )

    @pytest.mark.slow
    def test_doesnt_unnecessarily_truncate_short_sequences(self):
        transform = transforms.LinearizeTransform(
            tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            max_sequence_length=16,
            truncation_strategy="beginning",
        )
        # artificially fit the transform
        transform._fitted = True

        # when the first text is more than half the max length
        transformed = transform(
            ["A sentence that is more than 8 word pieces.", "Short."]
        )
        self.assertEqual(
            transformed["input_ids"],
            [
                0,
                250,
                3645,
                14,
                16,
                55,
                87,
                290,
                2136,
                3745,
                4,
                2,
                2,
                34256,
                4,
                2,
            ],
        )
        self.assertEqual(
            transformed["input_mask"],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        # when the second text is more than half the max length
        transformed = transform(
            ["Short.", "A sentence that is more than 8 word pieces."]
        )
        self.assertEqual(
            transformed["input_ids"],
            [
                0,
                34256,
                4,
                2,
                2,
                250,
                3645,
                14,
                16,
                55,
                87,
                290,
                2136,
                3745,
                4,
                2,
            ],
        )
        self.assertEqual(
            transformed["input_mask"],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        # when both texts are less than half the max length
        transformed = transform(["Short.", "Also short."])
        self.assertEqual(
            transformed["input_ids"],
            [0, 34256, 4, 2, 2, 22412, 765, 4, 2, 0, 0, 0, 0, 0, 0, 0],
        )
        self.assertEqual(
            transformed["input_mask"],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        )

    @pytest.mark.slow
    def test_truncates_the_longer_text_first(self):
        transform = transforms.LinearizeTransform(
            tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            max_sequence_length=12,
            truncation_strategy="beginning",
        )
        # artificially fit the transform
        transform._fitted = True

        # when the first text is longer
        transformed = transform(
            ["A sentence that is more than 8 word pieces.", "Short."]
        )
        self.assertEqual(
            transformed["input_ids"],
            [0, 250, 3645, 14, 16, 55, 87, 2, 2, 34256, 4, 2],
        )
        self.assertEqual(
            transformed["input_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )
        # when the second text is longer
        transformed = transform(
            ["Short.", "A sentence that is more than 8 word pieces."]
        )
        self.assertEqual(
            transformed["input_ids"],
            [0, 34256, 4, 2, 2, 250, 3645, 14, 16, 55, 87, 2],
        )
        self.assertEqual(
            transformed["input_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

    @pytest.mark.slow
    def test_truncates_texts_to_half_max_length_when_lots_of_text(self):
        transform = transforms.LinearizeTransform(
            tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            max_sequence_length=12,
            truncation_strategy="beginning",
        )
        # artificially fit the transform
        transform._fitted = True

        transformed = transform(
            [
                "This sentence is a test.",
                "A sentence that is more than 8 word pieces.",
            ]
        )
        self.assertEqual(
            transformed["input_ids"],
            [0, 713, 3645, 16, 10, 2, 2, 250, 3645, 14, 16, 2],
        )
        self.assertEqual(
            transformed["input_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

    @pytest.mark.slow
    def test_it_uses_the_correct_truncation_strategies(self):
        # test beginning strategy for first text
        transform = transforms.LinearizeTransform(
            tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            max_sequence_length=11,
            truncation_strategy=("beginning", "beginning"),
        )
        # artificially fit the transform
        transform._fitted = True

        transformed = transform(
            ["A sentence that is more than 8 word pieces.", "Short."]
        )
        self.assertEqual(
            transformed["input_ids"],
            [0, 250, 3645, 14, 16, 55, 2, 2, 34256, 4, 2],
        )
        self.assertEqual(
            transformed["input_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

        # test ending strategy for first text
        transform = transforms.LinearizeTransform(
            tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            max_sequence_length=11,
            truncation_strategy=("ending", "beginning"),
        )
        # artificially fit the transform
        transform._fitted = True

        transformed = transform(
            ("A sentence that is more than 8 word pieces.", "Short.")
        )
        self.assertEqual(
            transformed["input_ids"],
            [0, 87, 290, 2136, 3745, 4, 2, 2, 34256, 4, 2],
        )
        self.assertEqual(
            transformed["input_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

        # test beginning strategy for second text
        transform = transforms.LinearizeTransform(
            tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            max_sequence_length=11,
            truncation_strategy=("beginning", "beginning"),
        )
        # artificially fit the transform
        transform._fitted = True

        transformed = transform(
            ("Short.", "A sentence that is more than 8 word pieces.")
        )
        self.assertEqual(
            transformed["input_ids"],
            [0, 34256, 4, 2, 2, 250, 3645, 14, 16, 55, 2],
        )
        self.assertEqual(
            transformed["input_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

        # test ending strategy for second text
        transform = transforms.LinearizeTransform(
            tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            max_sequence_length=11,
            truncation_strategy=("beginning", "ending"),
        )
        # artificially fit the transform
        transform._fitted = True

        transformed = transform(
            ("Short.", "A sentence that is more than 8 word pieces.")
        )
        self.assertEqual(
            transformed["input_ids"],
            [0, 34256, 4, 2, 2, 87, 290, 2136, 3745, 4, 2],
        )
        self.assertEqual(
            transformed["input_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )
