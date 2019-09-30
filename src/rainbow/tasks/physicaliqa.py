"""The PhysicalIQA task."""

from fairseq.tasks import register_task

from . import base


@register_task("physicaliqa")
class PhysicalIQATask(base.MultipleChoiceTask):
    """The PhysicalIQA task."""

    def load_dataset(self, split, combine=False, **kwargs):
        raise NotImplementedError
