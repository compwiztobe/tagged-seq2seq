from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from .tuple_dictionary import TupleDictionary

@register_task('tagged_translation')
class TaggedTranslationTask(TranslationTask):
  @staticmethod
  def add_args(parser):
    pass

  @classmethod
  def load_dictionary(cls, *paths):
    return TupleDictionary(*[super().load_dictionary(path) for path in paths])
