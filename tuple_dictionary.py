import io
import os
from math import prod # as product # requires python 3.8
from itertools import product # as direct_product
from functools import cached_property
from collections import Counter

import torch
from fairseq.data import Dictionary, data_utils
from fairseq.file_io import PathManager

class FactorDictionary(Dictionary):
  """
  just a normal fairseq.data.Dictionary but with bos, pad, and eos special symbols removed
  these special symbols will be handled at the tuple level, not separately for each factor
  """
  def __init__(self, *, unk="<unk>", extra_special_symbols=None):
    super().__init__(unk=unk, extra_special_symbols=extra_special_symbols)
    self.symbols, self.count = zip(*[
      [s,c] for s,c in zip(self.symbols,self.count)
      if s not in [self.bos_word, self.pad_word, self.eos_word]
    ])
    self.symbols = list(self.symbols)
    self.count = list(self.count)
    self.indices = {s: i for i, s in enumerate(self.symbols)}
    del self.bos_word, self.pad_word, self.eos_word
    self.bos_index = self.pad_index = self.eos_index = None
    self.unk_index = self.symbols.index(self.unk_word)
    self.nspecial = len(self.symbols)

class SpecialSymbolDictionary(Dictionary):
  """
  just a normal fairseq.data.Dictionary but with unk removed
  to be handled at the factor level
  """
  def __init__(self, *, bos="<s>", pad="<pad>", eos="</s>", extra_special_symbols=None):
    super().__init__(bos=bos, pad=pad, eos=eos, extra_special_symbols=extra_special_symbols)
    self.symbols, self.count = zip(*[
      [s,c] for s,c in zip(self.symbols,self.count)
      if s != self.unk_word
    ])
    self.symbols = list(self.symbols)
    self.count = list(self.count)
    self.indices = {s: i for i, s in enumerate(self.symbols)}
    del self.unk_word
    self.unk_index = None
    # not really needed, but just in case
    self.bos_index = self.symbols.index(self.bos_word)
    self.pad_index = self.symbols.index(self.pad_word)
    self.eos_index = self.symbols.index(self.eos_word)
    self.nspecial = len(self.symbols)

class TupleDictionary(Dictionary):
  """
  convert multiple dictionaries with their own indices
  into a single dictionary with factored indices
  """
  def __init__(self, sep="<&&&>", factors=None, dicts=None, special_dict=None, extra_special_symbols=None):
    assert factors is not None or dicts is not None
    self.sep = sep
    if dicts is None:
      dicts = [FactorDictionary() for _ in range(factors)]
    self.dicts = dicts
    self.special_dict = special_dict or SpecialSymbolDictionary(extra_special_symbols=extra_special_symbols)
    self.counts = Counter()
    special_symbols = [
      (w,)*len(dicts)
      for w in [self.special_dict.bos_word, self.special_dict.pad_word, self.special_dict.eos_word]
    ] + [tuple(d.unk_word for d in dicts)]
    self.bos_word, self.pad_word, self.eos_word, self.unk_word = [
      sep.join(w for w in special_symbol) for special_symbol in special_symbols
    ]
    self.bos_index = self.special_dict.bos_index
    self.pad_index = self.special_dict.pad_index
    self.eos_index = self.special_dict.eos_index

  @property
  def unk_index(self):
    # # hack to recognize unks on the first factor - not appropriate for preprocessing!
    # class FirstFactorInt(int):
    #   def __eq__(a, b):
    #     return self.factor_index(a)[0] == self.factor_index(b)[0]
    return self.compute_index(d.unk_index for d in self.dicts)

  @property
  def nspecial(self):
    return len(self.special_dict)

  # I think I've specifically implemented all instances where this might have been called
  # so this is in fact unneeded?
  # def __getattr__(self, attr):
  #   try:
  #     return tuple(getattr(d, attr) for d in self.dicts)
  #   except AttributeError:
  #     raise AttributeError("'%s' object has no attribute '%s'", (type(self), attr))

  def __eq__(self, other):
    return hasattr(other, 'dicts') and self.dicts == other.dicts \
      and hasattr(other, 'special_dict') and self.special_dict == other.special_dict

  def __getitem__(self, index, as_tuple=False):
    if index < self.nspecial:
      symbols = (self.special_dict[index],)*len(self.dicts)
    else:
      indices = self.factor_index(index)
      symbols = tuple(d.symbols[i] for d, i in zip(self.dicts, indices))
    if as_tuple:
      return symbols
    else:
      return self.sep.join(symbols)

  def __len__(self):
    return self.nspecial + prod(self.factors)

  def __contains__(self, syms, as_tuple=False):
    if not as_tuple:
      syms = [sym.split(self.sep) for sym in syms]
    if len(set(syms)) == 1 and syms[0] in self.special_dict.symbols:
      return True
    else:
      return all(sym in d.symbols for sym, d in zip(syms, self.dicts))

  @property
  def symbols(self):
    return [(s,)*len(self.dicts) for s in self.special_dict.symbols] \
      + list(product(*[d.symbols for d in self.dicts]))

  @property
  def count(self):
    return self.special_dict.count + [self.counts[sym] for sym in self.symbols[self.nspecial:]]

  @property
  def indices(self):
    return {symbol: self.index(symbol, as_tuple=True) for symbol in self.symbols}

  #####

  @property
  def factors(self):
    return tuple(len(d) for d in self.dicts)

  def factor_index(self, index):
    if index < self.nspecial:
      return tuple(index, *[-1] * (len(self.factors) - 1))
    return tuple(
      (index - self.nspecial) % prod(self.factors[i:]) // prod(self.factors[i+1:])
      for i in range(len(self.factors))
    )

  # vectorized tensor version
  def factor_indices(self, indices, for_embedding=False):
    special_indices = indices < self.nspecial

    def conditional_factor(t, i):
      factored_indices = (t - self.nspecial) % prod(self.factors[i:]) // prod(self.factors[i+1:])
      factored_indices[special_indices] = t[special_indices] if i == 0 else -1
      return factored_indices.T

    factored_indices = [conditional_factor(indices, i) for i in range(len(self.factors))]

    if for_embedding:
      return torch.stack([
        indices + torch.where(special_indices.T, 0, self.nspecial + sum(self.factors[:i]))
        for i, indices in enumerate(factored_indices)
      ]).T
    else:
      return torch.stack(factored_indices).T

  def compute_index(self, indices):
    return self.nspecial + sum(
      prod(self.factors[i+1:])*index
      for i, index in enumerate(indices)
    )

  #####

  def index(self, syms, as_tuple=False):
    """Returns the index of the specified symbol"""
    assert isinstance(syms, str) or as_tuple
    if not as_tuple:
      syms = tuple(syms.split(self.sep))
    assert isinstance(syms, tuple) and len(syms) == len(self.dicts)
    if len(set(syms)) == 1 and syms[0] in self.special_dict.symbols:
      return self.special_dict.index(syms[0])
    else:
      return self.compute_index(d.index(sym) for d, sym in zip(self.dicts, syms))

  def string(
    self,
    tensor,
    bpe_symbol=None,
    escape_unk=False,
    extra_symbols_to_ignore=None,
    unk_string=None,
    first_factor_only=True
  ):
    # using inherited logic on pair indices, giving sep-joined strings
    # but must keep the subword symbols so we can process them later
    sents = super().string(tensor,
      bpe_symbol=None,
      escape_unk=escape_unk,
      extra_symbols_to_ignore=extra_symbols_to_ignore,
      unk_string=self.sep.join((unk_string,)*len(self.dicts))
    )
    # hack to insert unk_string correctly on individual factors
    # this will be VERY buggy...
    if unk_string:
      for d in self.dicts:
        sents = sents.replace(d.unk_word, unk_string)
    # separate out the first factor and detonize for BLEU scoring
    if first_factor_only:
      sents = [
        " ".join(t.split(self.sep)[0] for t in sent.split())
        for sent in sents.split("\n")
      ]
      return "\n".join(
        data_utils.post_process(sent, bpe_symbol)
        for sent in sents
      )
    else:
      return sents

  def unk_string(self, escape=False, as_tuple=False):
    unk = tuple(d.unk_string(escape) for d in self.dicts)
    if as_tuple:
      return unk
    else:
      return self.sep.join(unk)

  def add_symbol(self, word, n=1, overwrite=False, as_tuple=False):
    if not as_tuple:
      word = tuple(word.split(self.sep))
    self.counts[word] += n
    if len(set(word)) == 1 and word[0] in self.special_dict.symbols:
      return self.special_dict.add_symbol(word[0], n=n, overwrite=overwrite)
    else:
      indices = tuple(d.add_symbol(w, n=n, overwrite=overwrite) for d, w in zip(self.dicts, word))
      return self.compute_index(indices)

  # this is in fact never called anywhere in the fairseq codebase, so I won't worry about it
  def update(self, new_dict):
    raise NotImplementedError

  def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
    # thresholding and nwords are best at applied only to the first factor
    # since our other factors are probably very small, complete symbol sets
    # like pos or ner tags, instead of open vocabs we need to truncate
    self.dicts[0].finalize(threshold, nwords, padding_factor=1)

    # however,
    # padding at the level of the first factor dict does not guarantee padding
    # of the full tuple vocab to a multiple of 8 if the special symbols are not a
    # multiple of 8, and anyway madeupwords in the token factor are multiplied by
    # the tag factor, so we have more extra tuples than we really need in the full
    # vocab - instead it is better pad at the level of special symbols, based on
    # how many are need to add to the existing tuple vocab plus special symbols to
    # get to a multiple of 8
    self.pad_to_multiple_(padding_factor)

  def pad_to_multiple_(self, padding_factor):
    padding = -len(self) % padding_factor
    special_dict_size = len(self.special_dict) + padding
    self.special_dict.pad_to_multiple_(padding_factor=special_dict_size)

  # inherited implementations of these from Dictionary will do just fine
  # since I've implemented self.{bos,pad,eos,unk}_index in the constructor
  # def bos(self):
  #   """Helper to get index of beginning-of-sentence symbol"""
  #   return self.bos_index

  # def pad(self):
  #   """Helper to get index of pad symbol"""
  #   return self.pad_index

  # def eos(self):
  #   """Helper to get index of end-of-sentence symbol"""
  #   return self.eos_index

  # def unk(self):
  #   """Helper to get index of unk symbol"""
  #   return self.unk_index

  @classmethod
  def load(cls, f):
    if isinstance(f, str):
      try:
        with open(PathManager.get_local_path(f), "r", encoding="utf-8") as fd:
          return cls.load(fd)
      except FileNotFoundError as fnfe:
        raise fnfe
      except UnicodeError:
        raise Exception(
          "Incorrect encoding detected in {}, please "
          "rebuild the dataset".format(f)
        )

    # read factor header, then separate factor dict lines into their own streams for loading
    try:
      header = f.readline().strip()
      assert header.startswith("# factors ")
      nspecial = int(header.split()[-1])
      factors = tuple(int(f) for f in header.split()[2:-1]) # header.remove_prefix("# factors ").split()
      header = f.readline().strip()
      assert header.startswith("# sep ")
      sep = header.split()[-1] # header.remove_prefix("# sep ")
      assert nspecial + len(factors) > 0
      lines = f.readlines()
      if nspecial > 0:
        lines, special_lines = lines[:-nspecial], lines[-nspecial:]
        assert len(special_lines) == nspecial
        special_symbols = io.StringIO(''.join(special_lines))
      else:
        special_symbols = io.StringIO()
      assert len(lines) == sum(factors)
      factor_dicts = [
        io.StringIO(''.join(lines[sum(factors[:i]):sum(factors[:i+1])]))
        for i in range(len(factors))
      ]
    except (AssertionError, ValueError):
      raise ValueError(
        "Incorrect dictionary format, expected"
        "'# factor [factors] [special_symbol_count]' and "
        "'# sep [sep]',"
        "followed by that many lines for special symbols and each factor dictionary."
      )

    special_dict = SpecialSymbolDictionary.load(special_symbols)
    dicts = [FactorDictionary.load(factor_dict) for factor_dict in factor_dicts]
    return cls(sep, dicts=dicts, special_dict=special_dict)

  def save(self, f):
    """Stores dictionary into a text file
        a header line lists the factor dict sizes
        then factor dicts are concatenated in order
    """
    if isinstance(f, str):
      PathManager.mkdirs(os.path.dirname(f))
      with PathManager.open(f, "w", encoding="utf-8") as fd:
        return self.save(fd)

    all_dicts = self.dicts + [self.special_dict]
    # print a header that will throw an error if we try to load as a normal dict
    header = "# factors " + " ".join(str(len(d) - d.nspecial) for d in all_dicts)
    print(header, file=f)
    header = "# sep " + self.sep
    print(header, file=f)

    for d in all_dicts:
      d.save(f)

  def dummy_sentence(self, length):
    ts = [d.dummy_sentence(length) for d in self.dicts]
    t = torch.Tensor([self.compute_index(t) for t in zip(*ts)])
    t[-1] = self.eos()
    return t

  # inherited implementation works fine, with add_symbol and index implemented correctly
  # def encode_line(self):
  #   raise NotImplementedError

  # these two methods are unneeded, because with encode_line and such
  # properly implemented here, the static methods on Dictionary being called from
  # LegacyFairseqTask will do just fine
  @staticmethod
  def _add_file_to_dictionary_single_worker(
    filename, tokenize, eos_word, worker_id=0, num_workers=1
  ):
    raise NotImplementedError

  @staticmethod
  def add_file_to_dictionary(filename, dict, tokenize, num_workers):
    raise NotImplementedError
