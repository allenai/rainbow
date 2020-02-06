"""Mixture definitions for rainbow."""

import t5

from . import core, tasks

# N.B. tasks must be imported before mixtures, so that the mixtures can use
# the tasks in their definitions.


# AlphaNLI

t5.data.MixtureRegistry.add("anli_mixture", ["anli_task"], default_rate=1.0)
"""The AlphaNLI mixture."""


# CosmosQA

t5.data.MixtureRegistry.add(
    "cosmosqa_mixture", ["cosmosqa_task"], default_rate=1.0
)
"""The CosmosQA mixture."""


# HellaSWAG

t5.data.MixtureRegistry.add(
    "hellaswag_mixture", ["hellaswag_task"], default_rate=1.0
)
"""The HellaSWAG mixture."""


# PhysicalIQA

t5.data.MixtureRegistry.add(
    "physicaliqa_mixture", ["physicaliqa_task"], default_rate=1.0
)
"""The PhysicalIQA mixture."""


# SocialIQA

t5.data.MixtureRegistry.add(
    "socialiqa_mixture", ["socialiqa_task"], default_rate=1.0
)
"""The SocialIQA mixture."""


# WinoGrande

t5.data.MixtureRegistry.add(
    "winogrande_mixture", ["winogrande_task"], default_rate=1.0
)
"""The WinoGrande mixture."""


# Rainbow

t5.data.MixtureRegistry.add(
    "rainbow_equal_mixture",
    [
        "anli_task",
        "cosmosqa_task",
        "hellaswag_task",
        "physicaliqa_task",
        "socialiqa_task",
        "winogrande_task",
    ],
    default_rate=core.equal_rate,
)
"""The rainbow datasets mixed in equal amounts."""


t5.data.MixtureRegistry.add(
    "rainbow_proportional_mixture",
    [
        "anli_task",
        "cosmosqa_task",
        "hellaswag_task",
        "physicaliqa_task",
        "socialiqa_task",
        "winogrande_task",
    ],
    default_rate=core.proportional_rate,
)
"""The rainbow datasets mixed in proportional amounts."""
