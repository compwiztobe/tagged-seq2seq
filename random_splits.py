#!/usr/bin/env python

import random, sys

args = sys.argv[1:]

prefix = args[0]
args = args[1:]

dataset_size = int(args[0])
args = args[1:]

try:
  skip = int(args[0])
  dataset_size //= skip
  args = args[1:]
except ValueError:
  skip = 1

lengths = [int(split.split(',')[1]) for split in args]
names = [split.split(',')[0] for split in args]
split_files = [open(prefix+name,'x') for name in names]

splits = [x for i, length in enumerate(lengths) for x in [i]*length]
indices = random.sample(list(range(dataset_size)), sum(lengths))
index_map = {i: split for i, split in zip(indices, splits)}

#splits = []
#total_traversed = 0
#for length in lengths:
#  split =
#index_map = {i: name for name, split in zip(names, splits) for i in split}


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

if i + 1 < dataset_size:
  print("WARNING: only read %d lines out of %d expected" % (i + 1, dataset_size))

split_offset = 0
for name, length in zip(names, lengths):
  split = indices[split_offset:split_offset + length]
  with open(prefix+name+'.idx','x') as f:
    f.writelines(str(x)+'\n' for x in sorted(split))
  split_offset += length

print("%d lines read (in groups of %d)" % (i + 1, skip))
print("%d splits generated:" % len(lengths))
for name, length in zip(names, lengths):
  print("%s : %d lines (%d groups)" % (prefix + name, length * skip, length))
print("split indices written to {split}.idx")
print("total size of all splits: %d lines (%d groups)" % (sum(lengths) * skip, sum(lengths)))
