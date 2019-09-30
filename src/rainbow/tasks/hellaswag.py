"""The HellaSWAG task."""

from fairseq.tasks import register_task

from . import base


@register_task("hellaswag")
class HellaSWAG(base.MultipleChoiceTask):
    """The HellaSWAG task."""

    def load_dataset(self, split, combine=False, **kwargs):
        raise NotImplementedError
