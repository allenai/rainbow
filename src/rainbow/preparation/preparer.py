"""An abstract base class for dataset preparers."""

import abc


class Preparer(abc.ABC):
    """The abstract base class for a dataset preparer."""

    @abc.abstractmethod
    def prepare(self, src: str, dst: str, force_download: bool = False) -> None:
        """Preprocess and format all relevant datasets as text-to-text.

        Preprocess and format all relevant datasets, reading them from
        ``src`` and writing the results to ``dst``. If the datasets are
        not present in ``src``, then they will be automatically
        downloaded to ``src``.

        Google storage paths are supported.

        Parameters
        ----------
        src : str, required
            The source directory from which to read the datasets.
        dst : str, required
            The destination directory to which to write the prepared
            text-to-text datasets.
        force_download : bool, optional (default=False)
            If ``True``, then re-download all datasets to ``src``,
            regardless of whether or not files already exist at those
            paths.
        """
        raise NotImplementedError
