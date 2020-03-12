"""Mixing rates."""

import t5


# main functions


def equal_rate(task: t5.data.Task):
    """Mix the datasets in equal amounts.

    Parameters
    ----------
    task : t5.data.Task
        The task.

    Returns
    -------
    float
        The constant: ``1.0``.
    """
    return 1.0


def proportional_rate(task: t5.data.Task):
    """Mix the datasets proportionally.

    Parameters
    ----------
    task : t5.data.Task
        The task.

    Returns
    -------
    float
        The number of examples in the task's training set.
    """
    return float(task.num_input_examples("train"))


# constants

MIXING_RATES = {
    "equal": equal_rate,
    "proportional": proportional_rate,
}
"""A dictionary mapping mixing rates' names to their implementations."""
