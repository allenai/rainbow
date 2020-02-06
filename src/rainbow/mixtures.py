"""Mixture definitions for rainbow."""

import t5

from . import tasks

# N.B. tasks must be imported before mixtures, so that the mixtures can use
# the tasks in their definitions.


# AlphaNLI

t5.data.MixtureRegistry.add("anli", ["anli"], default_rate=1.0)
"""The AlphaNLI mixture."""


# CosmosQA

t5.data.MixtureRegistry.add("cosmosqa", ["cosmosqa"], default_rate=1.0)
"""The CosmosQA mixture."""


# HellaSWAG

t5.data.MixtureRegistry.add("hellaswag", ["hellaswag"], default_rate=1.0)
"""The HellaSWAG mixture."""


# PhysicalIQA

t5.data.MixtureRegistry.add("physicaliqa", ["physicaliqa"], default_rate=1.0)
"""The PhysicalIQA mixture."""


# SocialIQA

t5.data.MixtureRegistry.add("socialiqa", ["socialiqa"], default_rate=1.0)
"""The SocialIQA mixture."""


# WinoGrande

t5.data.MixtureRegistry.add("winogrande", ["winogrande"], default_rate=1.0)
"""The WinoGrande mixture."""
