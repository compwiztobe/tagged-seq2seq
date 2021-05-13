import sys,itertools

sep = sys.argv[1]
files = sys.argv[2:]

def read_vocab(f):
  with open(f) as f:
    return [line.strip().split()[0] for line in f.readlines()]

vocabs = [read_vocab(f) for f in files]

for tuple in itertools.product(*vocabs):
  print(sep.join(tuple))
