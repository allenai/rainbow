"""Utilities for rainbow."""

import contextlib
import hashlib
import logging

import attr

from . import settings


def configure_logging(verbose: bool = False) -> logging.Handler:
    """Configure logging and return the log handler.

    This function is useful in scripts when logging should be set up
    with a basic configuration.

    Parameters
    ----------
    verbose : bool, optional (default=False)
        If ``True`` set the log level to DEBUG, else set it to INFO.

    Returns
    -------
    logging.Handler
        The log handler set up by this function.
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


class FileLogging(contextlib.AbstractContextManager):
    """A context manager for logging to a file.

    Use this context manager to log output to a file. The context manager
    returns the log handler for the file. The manager attaches the handler to
    the root logger on entering the context and detaches it upon exit.

    Parameters
    ----------
    file_path : str, required
        The path at which to write the log file.

    Example
    -------
    Use the context manager as follows::

        with FileLogging('foo.log') as log_handler:
            # modify the log hander if desired.
            ...

    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.handler = None

    def __enter__(self) -> logging.Handler:
        if logging.root.level != logging.NOTSET:
            raise ValueError(
                "The root logger must have log level NOTSET to use the"
                " FileLogging context."
            )

        # Create the log handler for the file.
        handler = logging.FileHandler(self.file_path)

        # Configure the log handler.
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))

        # Attach the log handler to the root logger.
        logging.root.addHandler(handler)

        self.handler = handler

        return self.handler

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        logging.root.removeHandler(self.handler)


def string_to_seed(s: str) -> int:
    """Return an integer suitable for using as a random seed.

    Given a string, ``s``, return an integer suitable for using as a
    random seed in a deterministic way based on ``s``.

    Parameters
    ----------
    s : str, required
        The string to convert into an integer.
    """
    checksum = hashlib.sha256(s.encode("utf-8")).hexdigest()
    seed = int(checksum[:8], 16)
    return seed


def data(func):
    """A decorator for defining data."""
    return attr.s(auto_attribs=True, frozen=True, kw_only=True, slots=True)(
        func
    )
