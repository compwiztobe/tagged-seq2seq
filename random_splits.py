#!/usr/bin/env python

import argparse, random, sys

parser = argparse.ArgumentParser(description="Read from stdin and generate random splits given max index and split sizes")
parser.add_argument("prefix", help="prefix to the output file paths, built from split names")
parser.add_argument("splits", nargs="+", type=lambda s: (s.split(',')[0],int(s.split(',')[1])),
                    help="splits to generate, each as a split name followed by the number of skip-size groups in that split, separate by a comma")
parser.add_argument("--dataset-size", default=None, type=int, help="total skip-size group count from stdin to include in split coverage")
parser.add_argument("--reuse-indices", default=None, help="prefix to look for existing index files and use those (not random indices)")
parser.add_argument("--skip", default=1, type=int, help="how many lines to group together when selecting indices for each split")
parser.add_argument("--consecutive", action="store_true", help="do not randomize splits, take them consecutively from input")

args = parser.parse_args()

if not args.reuse_indices and not args.dataset_size:
  raise argparse.ArgumentError("either --dataset-size or --reuse-indices is required")

prefix = args.prefix
dataset_size = args.dataset_size
skip = args.skip
if dataset_size:
  dataset_size //= skip
names = [s[0] for s in args.splits]
lengths = [s[1] for s in args.splits]

split_files = [open(prefix+name,'x') for name in names]

def readlines(f):
  with open(f) as f:
    return f.readlines()

if args.reuse_indices:
  indices = [int(line.strip()) for name in names for line in readlines(args.reuse_indices+name+'.idx')]
elif args.consecutive:
  indices = list(range(sum(lengths)))
else:
  indices = random.sample(list(range(dataset_size)), sum(lengths))

splits = [x for i, length in enumerate(lengths) for x in [i]*length]
index_map = {i: split for i, split in zip(indices, splits)}


from contextlib import ExitStack

with ExitStack() as stack:
  for split_file in split_files:
    stack.enter_context(split_file)
  for i, line in enumerate(sys.stdin):
    index = i//skip
    if index in index_map:
      split_files[index_map[index]].write(line)

try:
  i
except NameError:
  print("nothing passed to stdin, quitting ...")
  sys.exit(1)

if dataset_size and i + 1 < dataset_size:
  print("WARNING: only read %d lines out of %d expected" % (i + 1, dataset_size))

split_offset = 0
for name, length in zip(names, lengths):
  split = indices[split_offset:split_offset + length]
  try:
    with open(prefix+name+'.idx','x') as f:
      f.writelines(str(x)+'\n' for x in sorted(split))
  except FileExistsError as e:
    if args.reuse_indices == prefix:
      print("index file %s already exists and was our input, so skipping write" % args.reuse_indices+name+'.idx')
    else:
      raise e
  split_offset += length

print("%d lines read (in groups of %d)" % (i + 1, skip))
print("%d splits generated:" % len(lengths))
for name, length in zip(names, lengths):
  print("%s : %d lines (%d groups)" % (prefix + name, length * skip, length))
print("split indices written to {split}.idx")
print("total size of all splits: %d lines (%d groups)" % (sum(lengths) * skip, sum(lengths)))
