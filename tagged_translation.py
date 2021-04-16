import os
import logging
import numpy as np

from fairseq import metrics, tokenizer, utils
from fairseq.data import Dictionary, data_utils
from fairseq.file_io import PathManager
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from .tuple_dictionary import TupleDictionary

logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4

@register_task('tagged_translation')
class TaggedTranslationTask(TranslationTask):
  @staticmethod
  def add_args(parser):
    TranslationTask.add_args(parser)

  @classmethod
  def setup_task(cls, args, **kwargs):
    # import traceback, sys
    # try:
    #   super().setup_task(args, **kwargs)
    # except AssertionError as e:
    #   logger.warning("WARNING: inherited task setup failed an assertion:", e)
    #   traceback.print_exception(*sys.exc_info())
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]
            )
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        # assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

  @classmethod
  def load_dictionary(cls, filename):
    """Load the dictionary from the filename

    Args:
      filename (str): the filename
    """
    return TupleDictionary.load(filename)

  @classmethod
  def build_dictionary(
    cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=1
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

  # def valid_step(self, sample, model, criterion):
  #   loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
  #   # if self.args.eval_F1:
  #   #   F1 = self.compute_F1(...) # this has a problem, already the hypothesis is generated
  #   #                             # in super().valid_step() -> _inference_with_bleu
  #   #                             # no need redoing that...
  #   #   logging_output['???'] = ???
  #   return loss, sample_size, logging_output

  # def reduce_metrics(self, logging_outputs, criterion):
  #   super().reduce_metrics(logging_outputs, criterion)
  #   # if self.args.eval_F1:
  #   #   # add up F1 metrics
  #   #   pass

  def reduce_metrics(self, logging_outputs, criterion):
      super(TranslationTask, self).reduce_metrics(logging_outputs, criterion)
      if self.args.eval_bleu:

          def sum_logs(key):
              import torch
              result = sum(log.get(key, 0) for log in logging_outputs)
              if torch.is_tensor(result):
                result = result.cpu()
              return result

          counts, totals = [], []
          for i in range(EVAL_BLEU_ORDER):
              counts.append(sum_logs("_bleu_counts_" + str(i)))
              totals.append(sum_logs("_bleu_totals_" + str(i)))

          # print([[log.get("_bleu_totals_" + str(i),0) for log in logging_outputs] for i in range(EVAL_BLEU_ORDER)])

          if max(totals) > 0:
              # log counts as numpy arrays -- log_scalar will sum them correctly
              metrics.log_scalar("_bleu_counts", np.array(counts))
              metrics.log_scalar("_bleu_totals", np.array(totals))
              metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
              metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

              def compute_bleu(meters):
                  import inspect
                  import sacrebleu

                  fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                  if "smooth_method" in fn_sig:
                      smooth = {"smooth_method": "exp"}
                  else:
                      smooth = {"smooth": "exp"}
                  bleu = sacrebleu.compute_bleu(
                      correct=meters["_bleu_counts"].sum,
                      total=meters["_bleu_totals"].sum,
                      sys_len=meters["_bleu_sys_len"].sum,
                      ref_len=meters["_bleu_ref_len"].sum,
                      **smooth
                  )
                  return round(bleu.score, 2)

              metrics.log_derived("bleu", compute_bleu)

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
