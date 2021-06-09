import fairseq

import sys, os, importlib
from collections import Counter

args = sys.argv[1:]

user_dir = args[0]
args = args[1:]

prefix = args[0]
args = args[1:]

splits = args[0].split(',')
args = args[1:]

src_lang,tgt_lang = args[:2]
args = args[2:]

import_dir, module = os.path.split(user_dir)
sys.path.insert(0, import_dir)
user_dir = importlib.import_module(module, module)

src_dict = user_dir.tuple_dictionary.TupleDictionary.load('%s/dict.%s.txt' % (prefix, src_lang))
tgt_dict = user_dir.tuple_dictionary.TupleDictionary.load('%s/dict.%s.txt' % (prefix, tgt_lang))

def sum_dicts(*dicts):
  keys = set.union(*[set(d.keys()) for d in dicts])
  return {key: sum(d.get(key,0) for d in dicts) for key in keys}

def sum_counters(*dicts):
  keys = set.union(*[set(d.keys()) for d in dicts])
  return {key: sum((d.get(key,Counter()) for d in dicts), start=Counter()) for key in keys}

def split_stats(split,lang,d):
  dataset = fairseq.data.data_utils.load_indexed_dataset('%s/%s.%s-%s.%s' % (prefix, split, src_lang, tgt_lang, lang),d,None)

  token_dict,tag_dict = d.dicts

  unks = 0
  NEs = 0
  NE_unks = 0
  total = 0

  unk_tags = Counter()
  tag_counts = Counter()

  for t in dataset:
    factors = [f for f in d.factor_indices(t) if int(f[0]) != token_dict.eos()]
    # check tags on unk words, confirming we're not counting any non tags (unk or anything)
    unk_tags |= Counter(tag_dict[int(f[1])] for f in factors if f[0] == token_dict.unk())
    # also check different tag counts out of curiousity
    tag_counts |= Counter(tag_dict[int(f[1])] for f in factors)

    # measurement
    unks += sum(int(f[0]) == token_dict.unk() for f in factors)
    NEs += sum(int(f[1]) != tag_dict.index('O') for f in factors)
    NE_unks += sum(int(f[0]) == token_dict.unk() and int(f[1]) != tag_dict.index('O') for f in factors)
    total += len(factors)

  return {
      'unks': unks,
      'NEs': NEs,
      'NE_unks': NE_unks,
      'total': total
  }, {
    'unk_tags': unk_tags,
    'tag_counts': tag_counts
  }

src_stats,src_counters = zip(*[split_stats(split, src_lang, src_dict) for split in splits])
tgt_stats,tgt_counters = zip(*[split_stats(split, tgt_lang, tgt_dict) for split in splits])
src_stats = sum_dicts(*src_stats)
tgt_stats = sum_dicts(*tgt_stats)
src_counters = sum_counters(*src_counters)
tgt_counters = sum_counters(*tgt_counters)

print("Debug...")
print(src_counters)
print(tgt_counters)
print()

def add_percentages(d):
  return dict(d,
    unk_rate=d['unks']/d['total'],
    NE_rate=d['NEs']/d['total'],
    NE_unk_rate1=d['NE_unks']/d['total'],
    NE_unk_rate2=d['NE_unks']/d['NEs']
  )

print("Results:")
print(add_percentages(src_stats))
print(add_percentages(tgt_stats))

