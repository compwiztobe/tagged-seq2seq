from flair.models import SequenceTagger
from flair.data import Sentence

import sys
import datetime

BATCH_SIZE=625

def tag_sentences(tagger, sentences, progress=False, batch_size=BATCH_SIZE):
  def tag_batch(batch):
    tagger.predict(batch, mini_batch_size=batch_size)
    return batch
  batch = []
  token_count = 0
  for i, sentence in enumerate(sentences):
    batch.append(sentence)
    token_count += len(sentence.tokens)
    if len(batch) == 10000:
      if progress:
        print("%s read %d sentences ..." % (datetime.datetime.now().isoformat(), i+1), file=sys.stderr)
      yield from tag_batch(batch)
      batch = []
      token_count = 0
  if batch:
    yield from tag_batch(batch)

def is_NE(token):
  # flair 0.8.0
  if 'ner' in token.annotation_layers:
    return any(tag.value != 'O' for tag in token.annotation_layers['ner'])
  # flair 0.4.5
  #if 'ner' in token.tags:
  #  return token.tags['ner'].value != 'O'
  return False

def main(model_file, sentences, separator, batch_size=BATCH_SIZE, ner_stats=False, progress=False):
  tagger = SequenceTagger.load(args.model_file)

  token_count = 0
  sentence_count = 0
  NE_count = 0

  sentences = (Sentence(s) for s in sentences)

  for tagged_sentence in tag_sentences(tagger, sentences, progress=progress, batch_size=batch_size):
    yield ' '.join(t.text + separator + t.annotation_layers[tagger.tag_type][0].value for t in tagged_sentence.tokens)
    if ner_stats:
      token_count += len(tagged_sentence.tokens)
      sentence_count += 1
      NE_count += sum(is_NE(token) for token in tagged_sentence.tokens)

  if ner_stats:
    return {
      'sentence_count': sentence_count,
      'token_count': token_count,
      'NE_count': NE_count
    }


import argparse
import logging
parser = argparse.ArgumentParser("Load a Flair NER model and use it to tag line-wise sentence data from stdin")
parser.add_argument("-m", "--model-file", required=True,
                   help="path to Flair model file (e.g. final-model.pt)")
parser.add_argument("--batch-size", default=BATCH_SIZE, type=int,
                    help="tagging batch size, in sentences (to optimize GPU mem use per model and dataset)")
parser.add_argument("--ner-stats", action='store_true',# action=argparse.BooleanOptionalAction, # python>=3.9 only
                    help="print NER tag statistics to stderr upon completion") # - if used with --print-tagged, stats go to stderr, unless another file is specified for tag outputs")
parser.add_argument("--print", action='store_true',
                    help="print tokens with tags to stdout, in plain text format with separator")
parser.add_argument("--separator", default="<&&&>",
                    help="separator with which to join tokens and tags (to be parsed into a tuple by fairseq")
parser.add_argument("--no-progress", dest='progress', default=True, action='store_false',
                    help="disable progress msgs on stderr")

def print_StopIterator(g, file=None):
  stop = yield from g
  if stop:
    print(stop, file=file)

if __name__ == "__main__":
  logging.getLogger('flair').handlers[0].stream = sys.stderr

  args = parser.parse_args()
  sentences = (line.strip().split() for line in sys.stdin)
  output = print_StopIterator(
    main(
      args.model_file,
      sentences,
      args.separator,
      batch_size=args.batch_size,
      ner_stats=args.ner_stats,
      progress=args.progress
    ), file=sys.stderr
  )
  if args.print:
    for line in output:
      print(line)
  else:
    for _ in output:
      pass
