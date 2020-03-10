"""Utilities for preparing datasets."""

from urllib import request

import tensorflow as tf


def copy_url_to_gfile(url: str, file_path: str) -> None:
    """Copy the contents located at ``url`` to ``file_path``.

    Parameters
    ----------
    url : str, required
        The URL from which to copy.
    file_path : str, required
        The file path to which to write. Google Cloud Storage file paths
        are supported.

    Returns
    -------
    None.
    """
    with request.urlopen(url) as data, tf.io.gfile.GFile(
        file_path, "w"
    ) as f_out:
        f_out.write(data.read())
