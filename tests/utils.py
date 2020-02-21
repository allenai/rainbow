"""Utilities for writing tests."""

import pkg_resources


def copy_pkg_resource_to_disk(pkg: str, src: str, dst: str) -> None:
    """Copy a package resource to disk.

    Copy the resource from the package ``pkg`` at the path ``src`` to
    disk at the path ``dst``.

    Parameters
    ----------
    pkg : str
        The package holding the resource.
    src : str
        The source path for the resource in ``pkg``.
    dst : str
        The destination path for the resource on disk.

    Notes
    -----
    This function is primarily useful for testing code that requires
    resources to be written on disk, when those test fixtures are
    shipped in the package.
    """
    with pkg_resources.resource_stream(pkg, src) as src_file, open(
        dst, "wb"
    ) as dst_file:
        dst_file.write(src_file.read())
