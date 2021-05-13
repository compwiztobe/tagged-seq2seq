import sys

sep = sys.argv[1]
files = sys.argv[2:]

files = [open(f) for f in files]

from contextlib import ExitStack

with ExitStack() as stack:
  for file in files:
    stack.enter_context(file)
  for line_tuple in zip(*files):
    print(' '.join(sep.join(tuple) for tuple in zip(*[line.strip().split() for line in line_tuple])))
      

