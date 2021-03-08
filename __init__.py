import logging, argparse
from . import tagged_translation, tagged_transformer

class ArgsHackHandler(logging.Handler):
  def handle(self, record):
    if isinstance(record, argparse.Namespace):
      tagged_translation.TaggedTranslationTask.sep = record.sep

logger = logging.getLogger("fairseq_cli.preprocess")
logger.addHandler(ArgsHackHandler())
print(logger.propagate)
print(logger.handlers[0].__dict__.keys())
