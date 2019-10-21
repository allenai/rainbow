"""Settings for rainbow."""

# feature augmentation constants

AUGMENTATION_TYPES = [
    "original",
    "atomic_text",
    "atomic_vector",
    "conceptnet_text",
    "conceptnet_vector",
]
"""The types of feature augmentation available."""


ATOMIC_RELATIONS = [
    "o_effect",
    "o_react",
    "o_want",
    "x_attr",
    "x_effect",
    "x_intent",
    "x_need",
    "x_react",
    "x_want",
]
"""The ATOMIC relations."""


CONCEPTNET_RELATIONS = [
    "at_location",
    "capable_of",
    "causes",
    "causes_desire",
    "created_by",
    "defined_as",
    "desire_of",
    "desires",
    "has_a",
    "has_first_subevent",
    "has_last_subevent",
    "has_pain_character",
    "has_pain_intensity",
    "has_prerequisite",
    "has_property",
    "has_subevent",
    "inherits_from",
    "instance_of",
    "is_a",
    "located_near",
    "location_of_action",
    "made_of",
    "motivated_by_goal",
    "not_capable_of",
    "not_desires",
    "not_has_a",
    "not_has_property",
    "not_is_a",
    "not_made_of",
    "part_of",
    "receives_action",
    "related_to",
    "symbol_of",
    "used_for",
]
"""The ConceptNet relations."""


# Logging and output

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
"""The format string for logging."""

TQDM_KWARGS = {"ncols": 72, "leave": False}
"""Key-word arguments for tqdm progress bars."""
