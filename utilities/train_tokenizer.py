import argparse
import sentencepiece

parser = argparse.ArgumentParser(prog="spm_train_light",
                                 description=("Command line interface for sentencepiece.SentencePieceTrainer.Train "
                                              "from sentencepiece pip package - options found at "
                                              "https://github.com/google/sentencepiece/blob/master/doc/options.md"))
parser.add_argument("input", nargs="+")
parser.add_argument("--model-prefix", required=True)
parser.add_argument("--vocab-size", type=int, default=8000)
parser.add_argument("--model-type", default="unigram", choices=["unigram", "bpe", "char", "word"])
parser.add_argument("--character-coverage", default=1.0)

def main(args):
  sentencepiece.SentencePieceTrainer.Train(**args.__dict__)

if __name__ == "__main__":
  main(parser.parse_args())
