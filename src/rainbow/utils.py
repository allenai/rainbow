"""Miscellaneous utilities."""

import logging
from typing import Any, Dict, List

from . import settings


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


def configure_logging(verbose: bool = False) -> logging.Handler:
    """Configure logging and return the log handler.

    This function is useful in scripts when logging should be set up
    with a basic configuration.

    Parameters
    ----------
    verbose : bool, optional (default=False)
        If ``True``, set the log level to DEBUG, else set the log level
        to INFO.

    Returns
    -------
    logging.Handler
        The log handler set up by this function to handle basic logging.
    """
    # unset the log level from root (defaults to WARNING)
    logging.root.setLevel(logging.NOTSET)

    # set up the log handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))

    # attach the log handler to root
    logging.root.addHandler(handler)

    return handler
