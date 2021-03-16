import os
import logging

from fairseq import tokenizer, utils
from fairseq.data import Dictionary
from fairseq.file_io import PathManager
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from .tuple_dictionary import TupleDictionary

logger = logging.getLogger(__name__)

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

  def valid_step(self, sample, model, criterion):
    loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
    # if self.args.eval_F1:
    #   F1 = self.compute_F1(...) # this has a problem, already the hypothesis is generated
    #                             # in super().valid_step() -> _inference_with_bleu
    #                             # no need redoing that...
    #   logging_output['???'] = ???
    return loss, sample_size, logging_output

  def reduce_metrics(self, logging_outputs, criterion):
    super().reduce_metrics(logging_outputs, criterion)
    # if self.args.eval_F1:
    #   # add up F1 metrics
    #   pass

  def _inference_with_bleu(self, generator, sample, model):
    import sacrebleu

    def decode(toks, escape_unk=False):
      s = self.tgt_dict.string(
        toks.int().cpu(),
        self.args.eval_bleu_remove_bpe,
          # The default unknown string in fairseq is `<unk>`, but
          # this is tokenized by sacrebleu as `< unk >`, inflating
          # BLEU scores. Instead, we use a somewhat more verbose
          # alternative that is unlikely to appear in the real
          # reference, but doesn't get split into multiple tokens.
          unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
          )
      if self.tokenizer:
        s = self.tokenizer.decode(s)
      return s

    gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
    hyps, refs = [], []
    for i in range(len(gen_out)):
      hyps.append(decode(gen_out[i][0]["tokens"]))
      refs.append(
        decode(
          utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
            escape_unk=True,  # don't count <unk> as matches to the hypo
            )
        )
    if self.args.eval_bleu_print_samples:
      logger.info("example hypothesis: " + hyps[0])
      logger.info("example reference: " + refs[0])

    hyps = [' '.join(pair.split(self.tgt_dict.sep)[0] for pair in hyp.split()) for hyp in hyps]
    refs = [' '.join(pair.split(self.tgt_dict.sep)[0] for pair in ref.split()) for ref in refs]

    if self.args.eval_tokenized_bleu:
      return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
    else:
      return sacrebleu.corpus_bleu(hyps, [refs])
