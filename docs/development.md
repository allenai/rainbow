Development
===========
Development documentation for `rainbow`.


Setup
-----
This project requires Python 3.6 or above.

First, install the project's dependencies:

    ./bin/install

Next, make sure you have the following environment variables set:

  1. `RAINBOW_DATASETS_DIR`: The directory for storing the rainbow
     datasets.
  2. `RAINBOW_PREPROCESSED_DATASETS_DIR`: The directory for storing the
     preprocessed dataset split files for rainbow.

Training requires TPUs. For training, all directories should point to Google
Cloud Storage prefixes. Additionally, you'll need the following environment
variables:

  1. `PROJECT`: Your Google Cloud project's ID.
  2. `ZONE`: Your Google Cloud virtual machine's zone.
  3. `TPU_NAME`: Your TPU's name.
  4. `TPU_TOPOLOGY`: Your TPU's topology.

Then, download and preprocess the rainbow datasets:

    $ ./bin/preprocess.py --help
    Usage: preprocess.py [OPTIONS]

      Preprocess the rainbow datasets for text-to-text modeling.

      Download to and read the rainbow datasets from --src, transform them into
      CSVs suitable for text-to-text models, then write the results to --dst.
      Google storage paths are supported.

    Options:
      --src TEXT        The directory to which to download the rainbow datasets.
                        Defaults to the RAINBOW_DATASETS_DIR environment variable.
                        [required]
      --dst TEXT        The directory to which to write the preprocessed dataset
                        files. Defaults to the RAINBOW_PREPROCESSED_DATASETS_DIR
                        environment variable.  [required]
      --force-download  Force downloads of all the datasets, otherwise only
                        missing datasets will be downloaded.
      --help            Show this message and exit.

Finally, verify your installation:

    ./bin/verify


Tests and Code Quality
----------------------
The code is formatted with [black][black]. You can run the formatter using the
[`bin/format`][bin/format] script:

    $ ./bin/format

To run code quality checks, use the [`bin/verify`][bin/verify] script:

    $ ./bin/verify

For fine-grained control of which tests to run, use [`pytest`][pytest]
directly:

    $ pytest

You can also skip slower tests by passing the `--skip-slow` (`-s`) flag:

    $ pytest --skip-slow


[black]: https://black.readthedocs.io/en/stable/
[bin/format]: ../bin/format
[bin/verify]: ../bin/verify
[pytest]: https://docs.pytest.org/en/latest/
