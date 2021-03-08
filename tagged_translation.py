import os

from fairseq import tokenizer
from fairseq.data import Dictionary
from fairseq.file_io import PathManager
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from .tuple_dictionary import TupleDictionary

@register_task('tagged_translation')
class TaggedTranslationTask(TranslationTask):
  @staticmethod
  def add_args(parser):
    TranslationTask.add_args(parser)

  @classmethod
  def load_dictionary(cls, filename):
    """Load the dictionary from the filename

    Args:
      filename (str): the filename
    """
    return TupleDictionary.load(filename)

  @classmethod
  def build_dictionary(
    cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
  ):
    """Build the dictionary

    Args:
      filenames (list): list of filenames
      workers (int): number of concurrent workers
      threshold (int): defines the minimum word count
      nwords (int): defines the total number of words in the final dictionary,
        including special symbols
      padding_factor (int): can be used to pad the dictionary size to be a
        multiple of 8, which is important on some hardware (e.g., Nvidia
        Tensor Cores).
            """

    # read sep from environment variable
    sep = os.environ['TAG_SEP'] # default to something?

    # read first tuple to determine factor count
    with open(PathManager.get_local_path(filenames[0]), "r", encoding="utf-8") as f:
      first_token = tokenizer.tokenize_line(f.readline())[0]
      factors = len(first_token.split(sep))

    d = TupleDictionary(sep, factors=factors)
    for filename in filenames:
      Dictionary.add_file_to_dictionary(
        filename, d, tokenizer.tokenize_line, workers
      )
    d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
    return d
