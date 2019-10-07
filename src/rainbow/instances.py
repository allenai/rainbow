"""Instance data types for various tasks."""

from typing import Dict, List

import attr

from .features import Feature


class Instance:
    """An abstract base class for instances."""

    pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class MultipleChoiceInstance(Instance):
    """A multiple choice instance."""

    features: Dict[str, Feature]
    answers: List[Feature]
    label: int
