#!/usr/bin/env python

import argparse
import random

parser = argparse.ArgumentParser("")
parser.add_argument("--naming-convention", default="prefix", values=["prefix", "suffix"])
parser.add_argument("--output-path", '-o', default="./")


def main(args):



def slice_iterator(iterator, indices):
  for i, item in enumerate(iterator):
    if i in indices:
      yield item

# kind of a funny signature for this function
# in all cases we require train < total_size
# split_indices(total_size, train) will give two splits,
#   the first of size train, and the second of size total_size - train
# split_indices(total_size, train, valid=?) or split_indices(total_size, train, test=?)
#   have the same behavior, requiring that valid/test add ups the remainder, and can therefore be ignored
# split_indices(total_size, train, valid=?, test=?) additionally requires they all add up
#   and will give three splits,
#   the first of size train, the second of size valid, and the third of size test, the remainder

def split_indices(total_size, train, valid=0, test=0):
  assert train < total_size and (train + valid + test == total_size or (valid == 0 and test == 0))
  indices = list(range(total_size))
  shuffled = random.shuffle(indices)
  if valid > 0 and test > 0:
    return shuffled[:train], shuffled[train:train+valid], shuffled[train+valid:train+valid+test]
  else:
    return shuffled[:train], shuffled[train:]

# a simpler approach (matching pytorch's torch.utils.data.random_split)
def random_split(dataset, lengths):
  shuffled = random.sample(dataset, len(dataset))
  splits = []
  total_traversed = 0
  for length in lengths:
    split = shuffled[total_traversed:total_traversed + length]
    total_traversed += length
    splits.append(split)
  return splits

if __name__ == '__main__':
  main(parser.parse_args())
