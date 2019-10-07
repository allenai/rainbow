"""Miscellaneous utilities."""

from typing import Any, Dict, List


def transpose_dictionary(d: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Return ``d`` as a list of dictionaries.

    Given a dictionary with values that are lists of the same length, transpose
    it into a list of dictionaries::

        >>> transpose_dictionary({'a': [1, 2, 3], 'b': [True, False, False]})
        [{'a': 1, 'b': True}, {'a': 2, 'b': False}, {'a': 3, 'b': False}]

    Parameters
    ----------
    d : Dict[str, List[Any]]
        A dictionary of lists all of the same length.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries representing the "transpose" of ``d``.
    """
    return [{k: v for k, v in zip(d.keys(), vs)} for vs in zip(*d.values())]
