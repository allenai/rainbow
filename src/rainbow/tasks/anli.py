"""The AlphaNLI task."""

from fairseq.tasks import register_task

from . import base


@register_task("anli")
class AlphaNLITask(base.MultipleChoiceTask):
    """The AlphaNLI task."""

    def load_dataset(self, split, combine=False, **kwargs):
        raise NotImplementedError
