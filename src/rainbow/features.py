"""Features."""

from typing import List

import attr


class Feature:
    """An abstract base class for features."""

    pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TextFeature(Feature):
    """A text feature."""

    text: str


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class CometAtomicFeature(Feature):
    """A text feature augmented with COMeT's ATOMIC generations."""

    text: str

    o_effect: str
    o_react: str
    o_want: str
    x_attr: str
    x_effect: str
    x_intent: str
    x_need: str
    x_react: str
    x_want: str

    o_effect_embeddings: List[List[int]]
    o_react_embeddings: List[List[int]]
    o_want_embeddings: List[List[int]]
    x_attr_embeddings: List[List[int]]
    x_effect_embeddings: List[List[int]]
    x_intent_embeddings: List[List[int]]
    x_need_embeddings: List[List[int]]
    x_react_embeddings: List[List[int]]
    x_want_embeddings: List[List[int]]


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class CometConceptNetFeature(Feature):
    """A text feature augmented with COMeT's ConceptNet generations."""

    text: str

    at_location: str
    capable_of: str
    causes: str
    causes_desire: str
    created_by: str
    defined_as: str
    desire_of: str
    desires: str
    has_a: str
    has_first_subevent: str
    has_last_subevent: str
    has_pain_character: str
    has_pain_intensity: str
    has_prerequisite: str
    has_property: str
    has_subevent: str
    inherits_from: str
    instance_of: str
    is_a: str
    located_near: str
    location_of_action: str
    made_of: str
    motivated_by_goal: str
    not_capable_of: str
    not_desires: str
    not_has_a: str
    not_has_property: str
    not_is_a: str
    not_made_of: str
    part_of: str
    receives_action: str
    related_to: str
    symbol_of: str
    used_for: str

    at_location_embeddings: List[List[int]]
    capable_of_embeddings: List[List[int]]
    causes_embeddings: List[List[int]]
    causes_desire_embeddings: List[List[int]]
    created_by_embeddings: List[List[int]]
    defined_as_embeddings: List[List[int]]
    desire_of_embeddings: List[List[int]]
    desires_embeddings: List[List[int]]
    has_a_embeddings: List[List[int]]
    has_first_subevent_embeddings: List[List[int]]
    has_last_subevent_embeddings: List[List[int]]
    has_pain_character_embeddings: List[List[int]]
    has_pain_intensity_embeddings: List[List[int]]
    has_prerequisite_embeddings: List[List[int]]
    has_property_embeddings: List[List[int]]
    has_subevent_embeddings: List[List[int]]
    inherits_from_embeddings: List[List[int]]
    instance_of_embeddings: List[List[int]]
    is_a_embeddings: List[List[int]]
    located_near_embeddings: List[List[int]]
    location_of_action_embeddings: List[List[int]]
    made_of_embeddings: List[List[int]]
    motivated_by_goal_embeddings: List[List[int]]
    not_capable_of_embeddings: List[List[int]]
    not_desires_embeddings: List[List[int]]
    not_has_a_embeddings: List[List[int]]
    not_has_property_embeddings: List[List[int]]
    not_is_a_embeddings: List[List[int]]
    not_made_of_embeddings: List[List[int]]
    part_of_embeddings: List[List[int]]
    receives_action_embeddings: List[List[int]]
    related_to_embeddings: List[List[int]]
    symbol_of_embeddings: List[List[int]]
    used_for_embeddings: List[List[int]]
