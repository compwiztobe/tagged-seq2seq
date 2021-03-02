from fairseq.tasks import TranslationTask, register_task

from tuple_dictionary import TupleDictionary

@register_task('tagged_translation')
class TaggedTranslationTask(TranslationTask):
  @staticmethod
  def add_args(parser):
    pass

  @classmethod
  def load_dictionary(cls, *paths):
    return TupleDictionary(*[super().load_dictionary(path) for path in paths])
