Development
===========
Information for developing `rainbow`.


Setup
-----
This project requires Python 3.6.

First, install the project's dependencies:

    ./bin/install

Then, verify your installation:

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
