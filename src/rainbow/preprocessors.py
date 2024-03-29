"""Preprocessors for modeling rainbow."""

from typing import Callable, Optional, Sequence

import tensorflow as tf


def make_add_field_names_preprocessor(
    field_names: Sequence[str], field_indices: Optional[Sequence[int]] = None,
) -> Callable:
    """Make a preprocessor to add field names to a dataset.

    Create a preprocessor that converts a dataset of lists of tensors
    into a dataset of dictionaries mapping strings to tensors.

    Parameters
    ----------
    field_names : Sequence[str], required
        A sequence of strings representing the field names for the new
        dictionaries.
    field_indices : Optional[Sequence[int]], optional (default=None)
        The indices corresponding to each field name in
        ``field_names``. If ``field_indices`` is ``None``, then each
        field name's corresponding index is assumed to be its index in
        the sequence.

    Returns
    -------
    Callable
        A function taking a ``tf.data.Dataset`` and returning a
        ``tf.data.Dataset``, that converts each sequence of tensors into an
        dictionary mapping the field names to the tensors at their
        corresponding indices.
    """
    if field_indices is None:
        field_indices = range(len(field_names))

    def add_field_names_preprocessor(
        dataset: tf.data.Dataset,
    ) -> tf.data.Dataset:
        return dataset.map(
            lambda *row: {
                field_name: row[field_index]
                for field_name, field_index in zip(field_names, field_indices)
            }
        )

    return add_field_names_preprocessor


def make_filter_preprocessor(predicate: Callable) -> Callable:
    """Make a preprocessor to filter examples from the dataset.

    Create a preprocessor that filters out any examples from the dataset
    for which the predicate returns ``False``.

    Parameters
    ----------
    predicate : Callable, required
        A function that takes an example and returns a boolean, ``True``
        if the example should remain in the dataset, ``False`` if it
        should not.

    Returns
    -------
    Callable
        A function taking a ``tf.data.Dataset`` and returning a
        ``tf.data.Dataset`` with all examples for which ``predicate``
        evaluates to ``False`` removed.
    """

    def filter_preprocessor(dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.filter(predicate)

    return filter_preprocessor
