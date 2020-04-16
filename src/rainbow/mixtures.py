"""Mixture definitions for rainbow."""

import t5

from . import datasets, rates, settings

import rainbow.tasks

# N.B. tasks must be imported before mixtures, so that the mixtures can use
# the tasks in their definitions.


# Create the individual rainbow task mixtures.
for dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        base_name = (
            f"{dataset.name}" if size is None else f"{dataset.name}_{size:05}"
        )
        t5.data.MixtureRegistry.add(
            f"{base_name}_mixture", [f"{base_name}_task"], default_rate=1.0
        )

# Create paired rainbow task mixtures.
for dataset1 in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for dataset2 in datasets.RAINBOW_DATASETS.values():
            for rate_name, rate_func in rates.MIXING_RATES.items():
                dataset1_base_name = (
                    f"{dataset1.name}"
                    if size is None
                    else f"{dataset1.name}_{size:05}"
                )
                t5.data.MixtureRegistry.add(
                    f"{dataset1_base_name}_{dataset2.name}_{rate_name}_mixture",
                    [f"{dataset1_base_name}_task", f"{dataset2.name}_task"],
                    default_rate=rate_func,
                )

# Create the individual knowledge graph task mixtures
for dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for direction in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
            base_name = (
                f"{dataset.name}_{direction}"
                if size is None
                else f"{dataset.name}_{direction}_{size:05}"
            )
            t5.data.MixtureRegistry.add(
                f"{base_name}_mixture", [f"{base_name}_task"], default_rate=1.0
            )

# Create the Rainbow mixtures.
for rate_name, rate_func in rates.MIXING_RATES.items():
    t5.data.MixtureRegistry.add(
        f"rainbow_{rate_name}_mixture",
        [
            f"{dataset.name}_task"
            for dataset in datasets.RAINBOW_DATASETS.values()
        ],
        default_rate=rate_func,
    )

# Create leave-one-out Rainbow mixtures.
for dataset in datasets.RAINBOW_DATASETS.values():
    for rate_name, rate_func in rates.MIXING_RATES.items():
        t5.data.MixtureRegistry.add(
            f"{dataset.name}_00000_rainbow_{rate_name}_mixture",
            [
                f"{other_dataset.name}_task"
                for other_dataset in datasets.RAINBOW_DATASETS.values()
                if other_dataset != dataset
            ],
            default_rate=rate_func,
        )

# Create knowledge graph-only mixtures with a single knowledge graph.
for rainbow_dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for kg_dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values():
            for direction in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
                for rate_name, rate_func in rates.MIXING_RATES.items():
                    t5.data.MixtureRegistry.add(
                        f"{rainbow_dataset.name}_{kg_dataset.name}_{direction}_{rate_name}_mixture"
                        if size is None
                        else f"{rainbow_dataset.name}_{size:05}_{kg_dataset.name}_{direction}_{rate_name}_mixture",
                        [
                            f"{rainbow_dataset.name}_task"
                            if size is None
                            else f"{rainbow_dataset.name}_{size:05}_task",
                            f"{kg_dataset.name}_{direction}_task",
                        ],
                        default_rate=rate_func,
                    )

# Create knowledge graph-only mixtures with all knowledge graphs.
for rainbow_dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for direction in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
            for rate_name, rate_func in rates.MIXING_RATES.items():
                t5.data.MixtureRegistry.add(
                    f"{rainbow_dataset.name}_comet_{direction}_{rate_name}_mixture"
                    if size is None
                    else f"{rainbow_dataset.name}_{size:05}_comet_{direction}_{rate_name}_mixture",
                    [
                        f"{rainbow_dataset.name}_task"
                        if size is None
                        else f"{rainbow_dataset.name}_{size:05}_task"
                    ]
                    + [
                        f"{kg_dataset.name}_{direction}_task"
                        for kg_dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values()
                    ],
                    default_rate=rate_func,
                )

# Create mixtures with rainbow and a single knowledge graph.
for dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values():
    for direction in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
        for rate_name, rate_func in rates.MIXING_RATES.items():
            t5.data.MixtureRegistry.add(
                f"rainbow_{dataset.name}_{direction}_{rate_name}_mixture",
                [f"{dataset.name}_{direction}_task"]
                + [
                    f"{rainbow_dataset.name}_task"
                    for rainbow_dataset in datasets.RAINBOW_DATASETS.values()
                ],
                default_rate=rate_func,
            )

# Create mixtures with rainbow and all knowledge graphs.
for direction in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
    for rate_name, rate_func in rates.MIXING_RATES.items():
        t5.data.MixtureRegistry.add(
            f"rainbow_comet_{direction}_{rate_name}_mixture",
            [
                # include the knowledge graphs
                f"{dataset.name}_{direction}_task"
                for dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values()
            ]
            + [
                # include the rainbow datasets
                f"{dataset.name}_task"
                for dataset in datasets.RAINBOW_DATASETS.values()
            ],
            default_rate=rate_func,
        )

# Create learning curve mixtures with rainbow and a single knowledge graph
for rainbow_dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        if size is None:
            continue

        for kg_dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values():
            for direction in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
                for rate_name, rate_func in rates.MIXING_RATES.items():
                    t5.data.MixtureRegistry.add(
                        f"{rainbow_dataset.name}_{size:05}_rainbow"
                        f"_{kg_dataset.name}_{direction}_{rate_name}_mixture",
                        [
                            f"{rainbow_dataset.name}_{size:05}_task",
                            f"{kg_dataset.name}_{direction}_task",
                        ]
                        + [
                            f"{other_dataset.name}_task"
                            for other_dataset in datasets.RAINBOW_DATASETS.values()
                            if other_dataset != rainbow_dataset
                        ],
                        default_rate=rate_func,
                    )

# Create learning curve mixtures with rainbow and all knowledge graphs
for rainbow_dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        if size is None:
            continue

        for direction in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
            for rate_name, rate_func in rates.MIXING_RATES.items():
                t5.data.MixtureRegistry.add(
                    f"{rainbow_dataset.name}_{size:05}_rainbow"
                    f"_comet_{direction}_{rate_name}_mixture",
                    [f"{rainbow_dataset.name}_{size:05}_task"]
                    + [
                        # include knowledge graphs
                        f"{dataset.name}_{direction}_task"
                        for dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values()
                    ]
                    + [
                        # include other rainbow tasks
                        f"{dataset.name}_task"
                        for dataset in datasets.RAINBOW_DATASETS.values()
                        if dataset != rainbow_dataset
                    ],
                    default_rate=rate_func,
                )

# Create the Rainbow multi-tasking learning curve mixtures.
for dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        if size is None:
            continue

        for rate_name, rate_func in rates.MIXING_RATES.items():
            t5.data.MixtureRegistry.add(
                f"{dataset.name}_{size:05}_rainbow_{rate_name}_mixture",
                [f"{dataset.name}_{size:05}_task"]
                + [
                    f"{other_dataset.name}_task"
                    for other_dataset in datasets.RAINBOW_DATASETS.values()
                    if other_dataset != dataset
                ],
                default_rate=rate_func,
            )

# Create the GLUE multi-tasking learning curve mixtures.
for dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for rate_name, rate_func in rates.MIXING_RATES.items():
            t5.data.MixtureRegistry.add(
                f"{dataset.name}_glue_{rate_name}_mixture"
                if size is None
                else f"{dataset.name}_{size:05}_glue_{rate_name}_mixture",
                [
                    f"{dataset.name}_task"
                    if size is None
                    else f"{dataset.name}_{size:05}_task"
                ]
                + [
                    f"glue_{glue_dataset.name}_v002"
                    for glue_dataset in datasets.GLUE_DATASETS.values()
                ],
                default_rate=rate_func,
            )

# Create the Super GLUE multi-tasking learning curve mixtures.
for dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for rate_name, rate_func in rates.MIXING_RATES.items():
            t5.data.MixtureRegistry.add(
                f"{dataset.name}_super_glue_{rate_name}_mixture"
                if size is None
                else f"{dataset.name}_{size:05}_super_glue_{rate_name}_mixture",
                [
                    f"{dataset.name}_task"
                    if size is None
                    else f"{dataset.name}_{size:05}_task"
                ]
                + [
                    f"super_glue_{super_glue_dataset.name}_v102"
                    for super_glue_dataset in datasets.SUPER_GLUE_DATASETS.values()
                ],
                default_rate=rate_func,
            )


# Create rainbow external evaluation mixtures.
for dataset in datasets.COMMONSENSE_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for rate_name, rate_func in rates.MIXING_RATES.items():
            base_name = (
                dataset.name if size is None else f"{dataset.name}_{size:05}"
            )
            t5.data.MixtureRegistry.add(
                f"{base_name}_rainbow_{rate_name}_mixture",
                [f"{base_name}_task"]
                + [
                    f"{rainbow_dataset.name}_task"
                    for rainbow_dataset in datasets.RAINBOW_DATASETS.values()
                ],
                default_rate=rate_func,
            )

# Create GLUE external evaluation mixtures.
for dataset in datasets.COMMONSENSE_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for rate_name, rate_func in rates.MIXING_RATES.items():
            base_name = (
                dataset.name if size is None else f"{dataset.name}_{size:05}"
            )
            t5.data.MixtureRegistry.add(
                f"{base_name}_glue_{rate_name}_mixture",
                [f"{base_name}_task"]
                + [
                    f"glue_{glue_dataset.name}_v002"
                    for glue_dataset in datasets.GLUE_DATASETS.values()
                ],
                default_rate=rate_func,
            )

# Create Super GLUE external evaluation mixtures.
for dataset in datasets.COMMONSENSE_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for rate_name, rate_func in rates.MIXING_RATES.items():
            base_name = (
                dataset.name if size is None else f"{dataset.name}_{size:05}"
            )
            t5.data.MixtureRegistry.add(
                f"{base_name}_super_glue_{rate_name}_mixture",
                [f"{base_name}_task"]
                + [
                    f"super_glue_{super_glue_dataset.name}_v102"
                    for super_glue_dataset in datasets.SUPER_GLUE_DATASETS.values()
                ],
                default_rate=rate_func,
            )
