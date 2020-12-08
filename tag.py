from flair.models import SequenceTagger
from flair.data import Sentence

import sys
import datetime

def tag_sentences(tagger, sentences):
  def tag_batch(batch, token_count):
    print("Tagging batch of %d sentences, %s tokens" % (len(batch), token_count), file=sys.stderr)
    tagger.predict(batch, mini_batch_size=256)
    return batch
  batch = []
  token_count = 0
  for i, sentence in enumerate(sentences):
    if args.progress and i % 1000 == 0:
      print("%s read %d sentences ..." % (datetime.datetime.now().isoformat(), i), file=sys.stderr)
    batch.append(sentence)
    token_count += len(sentence.tokens)
    if token_count >= 65536: # len(batch) == 1024 # or 1000 with batch size 250, and log on 1000
      yield from tag_batch(batch, token_count)
      batch = []
      token_count = 0
  if batch:
    yield from tag_batch(batch, token_count)

def is_NE(token):
  return any(tag.value != 'O' for tag in token.annotation_layers['ner'])

def main(args):
  print(args)
  # yield from range(10)
  # return 100
  tagger = SequenceTagger.load(args.model_file)

  token_count = 0
  sentence_count = 0
  NE_count = 0

  sentences = (Sentence(line) for line in sys.stdin if line.strip() != "" and not line.startswith("#"))

  for tagged_sentence in tag_sentences(tagger, sentences):
    if args.print_tags:
      yield tagged_sentence
    token_count += len(tagged_sentence.tokens)
    sentence_count += 1
    NE_count += sum(is_NE(token) for token in tagged_sentence.tokens)

  return {
    'sentence_count': sentence_count,
    'token_count': token_count,
    'NE_count': NE_count
  }


import argparse
parser = argparse.ArgumentParser("Load a Flair NER model and use it to tag line-wise sentence data from stdin")
parser.add_argument("-m", "--model-file", required=True,
                   help="path to Flair model file (e.g. final-model.pt)")
parser.add_argument("--stats", default=True, action='store_true',# action=argparse.BooleanOptionalAction, # python>=3.9 only
                    help="print NER tag statistics to stderr upon completion") # - if used with --print-tagged, stats go to stderr, unless another file is specified for tag outputs")
parser.add_argument("--print-tags", default=False, action='store_true',
                    help="print NER tags to stdout, in .bio (Flair input) format")
parser.add_argument("--no-progress", dest='progress', default=True, action='store_false',
                    help="disable progress msgs on stderr")
# parser.add_argument("--print-tags", metavar="TAG_FILE", default=False, nargs='?', const=True,
#                     help="print NER tags to file, or stdout if unspecified, in .bio format (Flair input format)")

def print_return(g, file=sys.stderr):
  stop = yield from g
  print(stop, file=file)

if __name__ == "__main__":
  args = parser.parse_args()
  output = print_return(main(parser.parse_args()))
  for line in output:
      print(line)
