from sentencepiece import SentencePieceProcessor as SentencePiece

import sys

def is_NE(token, tag_types):
  if len(token) > 1:
    return token[tag_types.index('ner')+1 if 'ner' in tag_types else 1] != 'O'
  return False

# tokenize a list of words into a flattened list of subwords with tags broadcasted
def tokenize(tokenizer, words, sep, tag_types):
  for w in words:
    word, *tags = w.split(sep)
    tokens = tokenizer.encode(word, out_type=str)
    if len(tags) > len(tag_types):
      print("WARNING: %d tags found but only %d --tag-types supplied" % (len(tags), len(tag_types)), file=sys.stderr)
    broadcast_tags = []
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
        broadcast_tags.append(
            [b_tag if tag.startswith("B-") or tag.startswith("S-") else i_tag]
          + [i_tag for _ in tokens[1:-1]]
          + [e_tag if tag.startswith("E-") or tag.startswith("S-") else i_tag]
        )
        # else: # other types of tags all get converted to B I E
        #   broadcast_tags.append(
        #       [b_tag]
        #     + [i_tag for _ in tokens[1:-1]]
        #     + [e_tag]
        #   )
      # broadcast duplicates for special cases (O for ner, X for pos)
      else:
        broadcast_tags.append([tag for _ in tokens])
    yield from zip(tokens, *broadcast_tags)

def main(args):
  tokenizer = SentencePiece(args.model_file)

  token_count = 0
  sentence_count = 0
  NE_count = 0

  for line in sys.stdin:
    tokenized = list(tokenize(tokenizer, line.strip().split(), args.separator, args.tag_types))
    print(" ".join(args.separator.join(t for t in token) for token in tokenized))
    if args.ner_stats:
      token_count += len(tokenized)
      sentence_count += 1
      NE_count += sum(is_NE(token, args.tag_types) for token in tokenized)

  if args.ner_stats:
    print({
      'sentence_count': sentence_count,
      'token_count': token_count,
      'NE_count': NE_count
    }, file=sys.stderr)

import argparse
parser = argparse.ArgumentParser(description="Load a SentencePiece tokenizer and process line-wise sentence data from stdin")
parser.add_argument("-m", "--model-file", required=True,
                    help="path to SentencePiece subword model file (e.g. tokens.model)")
parser.add_argument("--ner-stats", action='store_true',# action=argparse.BooleanOptionalAction, # python>=3.9 only
                    help="print NER tag statistics to stderr upon completion") # - if used with --print, stats go to stderr, unless another file is specified for tag outputs")
parser.add_argument("--separator", default="<&&&>",
                    help="separator with which to join tokens and tags (to be parsed into a tuple by fairseq")
parser.add_argument("--tag-types", nargs="*", default=[],
                    help="space separated list of tag types for each tag channel in the input - this determines how tags are broadcast across subwords")

if __name__ == "__main__":
  main(parser.parse_args())
