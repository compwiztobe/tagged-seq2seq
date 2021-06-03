from sentencepiece import SentencePieceProcessor as SentencePiece

import sys

def is_NE(token, tag_types):
  if len(token) > 1:
    return token[tag_types.index('ner')+1 if 'ner' in tag_types else 1] != 'O'
  return False

def tokenize(tokenizer, words, sep, tag_types):
  subwords = []
  for w in words:
    word, *tags = w.split(sep)
    tokens = tokenizer.encode(word, out_type=str)
    if len(tags) > len(tag_types):
      print("WARNING: %d tags found but only %d --tag-types supplied" % (len(tags), len(tag_types)), file=sys.stderr)
    for tag, tag_type in zip(tags, tag_types):
      # broadcast the tag across multiple subwords, if maintaining BIOES
      if len(tokens) > 1 and (
          (tag_type == 'ner' and '-' in tag)
        # or
        #   (tag_type == 'upos' and tag != 'X')
        ):
        # if tag_type == 'ner':
        b_tag = "B-" + tag.split("-", 1)[1]
        i_tag = "I-" + tag.split("-", 1)[1]
        e_tag = "E-" + tag.split("-", 1)[1]
        # if tag_type == 'ner': # ner tags need existing B I E respected, not divided
        broadcast_tags = \
            [b_tag if tag.startswith("B-") or tag.startswith("S-") else i_tag] \
          + [i_tag for _ in tokens[1:-1]] \
          + [e_tag if tag.startswith("E-") or tag.startswith("S-") else i_tag]
        # else: # other types of tags all get converted to B I E
        #   broadcast_tags = \
        #       [b_tag] \
        #     + [i_tag for _ in tokens[1:-1]] \
        #     + [e_tag]
      # broadcast duplicates for special cases (O for ner, X for pos)
      else:
        broadcast_tags = [tag for _ in tokens]
    subwords.extend([(t,tags) for t, tags in zip(tokens, broadcast_tags)])
  return subwords

def main(args):
  tokenizer = SentencePiece(args.model_file)

  token_count = 0
  sentence_count = 0
  NE_count = 0

  sentences = (line.strip().split() for line in sys.stdin)

  for sentence in sentences:
    tokenized = tokenize(tokenizer, sentence, args.separator, args.tag_types)
    if args.print:
      yield " ".join(args.separator.join(t for t in token) for token in tokenized)
    if args.ner_stats:
      token_count += len(tokenized)
      sentence_count += 1
      NE_count += sum(is_NE(token, args.tag_types) for token in tokenized)

  if args.ner_stats:
    return {
      'sentence_count': sentence_count,
      'token_count': token_count,
      'NE_count': NE_count
    }


import argparse
parser = argparse.ArgumentParser(description="Load a SentencePiece tokenizer and process line-wise sentence data from stdin")
parser.add_argument("-m", "--model-file", required=True,
                    help="path to SentencePiece subword model file (e.g. tokens.model)")
parser.add_argument("--ner-stats", action='store_true',# action=argparse.BooleanOptionalAction, # python>=3.9 only
                    help="print NER tag statistics to stderr upon completion") # - if used with --print, stats go to stderr, unless another file is specified for tag outputs")
parser.add_argument("--print", action='store_true',
                    help="print tokens with tags to stdout, in plain text format with separator")
parser.add_argument("--separator", default="<&&&>",
                    help="separator with which to join tokens and tags (to be parsed into a tuple by fairseq")
parser.add_argument("--tag-types", nargs="*", default=[],
                    help="space separated list of tag types for each tag channel in the input - this determines how tags are broadcast across subwords")
parser.add_argument("--no-progress", dest='progress', default=True, action='store_false',
                    help="disable progress msgs on stderr")

def print_StopIterator(g, file=None):
  stop = yield from g
  if stop:
    print(stop, file=file)

if __name__ == "__main__":
  args = parser.parse_args()
  output = print_StopIterator(main(parser.parse_args()), file=sys.stderr)
  for line in output:
    print(line)
