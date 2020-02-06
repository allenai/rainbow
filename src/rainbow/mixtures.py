"""Mixture definitions for rainbow."""

import t5

from . import tasks

# N.B. tasks must be imported before mixtures, so that the mixtures can use
# the tasks in their definitions.


t5.data.MixtureRegistry.add("socialiqa", ["socialiqa"], default_rate=1.0)
