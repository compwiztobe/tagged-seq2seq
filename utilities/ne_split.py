#!/usr/bin/env python

import random, sys

args = sys.argv[1:]

prefix = args[0]
args = args[1:]

sep = args[0]
args = args[1:]

try:
  skip = int(args[0])
  args = args[1:]
except ValueError:
  skip = 1

no_ne_lines = []
ne_lines = []
ne_src_lines = []
ne_tgt_lines = []

with open(prefix+'.noNE','x') as no_ne_split, open(prefix+'.someNE','x') as ne_split:
  for i, lines in enumerate(zip(*[sys.stdin]*skip)):
    tags = [[pair.split(sep)[1] for pair in line.strip().split()] for line in lines]
    has_ne = any(tag != "O" for line in tags for tag in line)
    has_ne_src = any(tag != "O" for tag in tags[0])
    has_ne_tgt = any(tag != "O" for tag in tags[1])
    assert has_ne == has_ne_src or has_ne_tgt
    if has_ne:
      ne_lines.append(i)
      ne_split.writelines(lines)
    else:
      no_ne_lines.append(i)
      no_ne_split.writelines(lines)
    if has_ne_src:
      ne_src_lines.append(i)
    if has_ne_tgt:
      ne_tgt_lines.append(i)

with open(prefix+'.noNE.idx','x') as no_ne_idx:
  no_ne_idx.writelines(str(i)+'\n' for i in no_ne_lines)
with open(prefix+'.someNE.idx','x') as ne_idx:
  ne_idx.writelines(str(i)+'\n' for i in ne_lines)

with open(prefix+'.NEsrc.idx','x') as ne_src_idx:
  ne_src_idx.writelines(str(i)+'\n' for i in ne_src_lines)
with open(prefix+'.NEtgt.idx','x') as ne_tgt_idx:
  ne_tgt_idx.writelines(str(i)+'\n' for i in ne_tgt_lines)
with open(prefix+'.NEdiff.idx','x') as ne_diff_idx:
  ne_diff_idx.writelines(str(i)+'\n' for i in set(ne_src_lines) ^ set(ne_tgt_lines))

print("%d found with no NEs" % len(no_ne_lines))
print("%d found with NEs" % len(ne_lines))
print("%d found with src NEs" % len(ne_src_lines))
print("%d found with tgt NEs" % len(ne_tgt_lines))
print("%d found with src or tgt NEs but not both" % len(set(ne_src_lines) ^ set(ne_tgt_lines)))
