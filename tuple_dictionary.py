import io
import os
from math import prod # as product # requires python 3.8
from itertools import product # as direct_product
from functools import cached_property
from collections import Counter

import torch
from fairseq.data import Dictionary
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


class TupleDictionary(Dictionary):
  """
  convert multiple dictionaries with their own indices
  into a single dictionary with factored indices
  """
  def __init__(self, sep="<&&&>", factors=None, dicts=None, extra_special_symbols=None):
    assert factors is not None or dicts is not None
    self.sep = sep
    if dicts is None:
      dicts = [FactorDictionary() for _ in range(factors)]
    self.dicts = dicts
    bos_word, pad_word, eos_word = ["<s>", "<pad>", "</s>"]
    self.special_symbols = []
    self.nspecial = 0
    self.counts = Counter()
    self.bos_word, self.pad_word, self.eos_word = [sep.join((w,)*len(dicts)) for w in [bos_word, pad_word, eos_word]]
    self.unk_word = sep.join(d.unk_word for d in dicts)
    self.bos_index = self.add_special_symbol(self.bos_word, as_tuple=True)
    self.pad_index = self.add_special_symbol(self.pad_word, as_tuple=True)
    self.eos_index = self.add_special_symbol(self.eos_word, as_tuple=True)
    if extra_special_symbols:
      for s in extra_special_symbols:
        self.add_special_symbol(s, as_tuple=False)

  @property
  def unk_index(self):
    return self.compute_index(d.unk_index for d in self.dicts)

  # I think I've specifically implemented all instances where this might have been called
  # so this is in fact unneeded?
  # def __getattr__(self, attr):
  #   try:
  #     return tuple(getattr(d, attr) for d in self.dicts)
  #   except AttributeError:
  #     raise AttributeError("'%s' object has no attribute '%s'", (type(self), attr))

  def __eq__(self, other):
    return hasattr(other, 'dicts') and self.dicts == other.dicts

  def __getitem__(self, index, as_tuple=False):
    if index < self.nspecial:
      symbols = self.special_symbols[index]
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
    if syms in self.special_symbols:
      return True
    else:
      return all(sym in d.symbols for sym, d in zip(syms, self.dicts))

  #####

  @property
  def symbols(self):
    return self.special_symbols + list(product(*[d.symbols for d in self.dicts]))

  @property
  def count(self):
    return [self.counts[sym] for sym in self.symbols]

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

  # this should probably be constructed once upon dictionary finalize or something
  # to save some compute, instead of building it every time we want to embed a batch
  @cached_property
  def _factor_indicator_map(self):
    coords = torch.LongTensor([
      list(range(self.nspecial)) + [self.nspecial + row for row in range(prod(self.factors)) for _ in range(len(self.factors))],
      list(range(self.nspecial)) + [self.nspecial + x + sum(self.factors[:i]) for idx in product(*[range(n) for n in self.factors]) for i, x in enumerate(idx)]
    ])
    values = torch.ones(self.nspecial + prod(self.factors) * len(self.factors))
    size = torch.Size((self.nspecial + prod(self.factors), self.nspecial + sum(self.factors)))
    return coords, values, size

  @property
  def factor_indicator_map(self):
    args = self._factor_indicator_map
    return torch.sparse.LongTensor(*args).cuda()

  #####

  def index(self, syms, as_tuple=False):
    """Returns the index of the specified symbol"""
    assert isinstance(syms, str) or as_tuple
    if not as_tuple:
      syms = tuple(syms.split(self.sep))
    assert isinstance(syms, tuple) and len(syms) == len(self.dicts)
    if syms in self.special_symbols:
      return self.special_symbols.index(syms)
    else:
      return self.compute_index(d.index(sym) for d, sym in zip(self.dicts, syms))

  def string(
    self,
    tensor,
    bpe_symbol=None,
    escape_unk=False,
    extra_symbols_to_ignore=None,
    unk_string=None,
    factored_indices=False,
    as_tuple=False
  ):
    if torch.is_tensor(tensor) and tensor.dim() == 2 + factored_indices:
      return "\n".join(
        self.string(t, bpe_symbol, escape_unk, extra_symbols_to_ignore, unk_string, factored_indices, as_tuple)
        for t in tensor
      )

    if not factored_indices:
      tensor = self.factor_indices(tensor)
    strings = [
      [
        d.string([index], bpe_symbol, escape_unk, extra_symbols_to_ignore, unk_string)
        for index in indices if index >= 0
      ]
      for d, indices in zip(self.dicts, zip(*tensor))
    ]
    if as_tuple:
      return " ".join(str(t) for t in zip(*strings))
    else:
      return " ".join(self.sep.join(t) for t in zip(*strings))

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
    if word in self.special_symbols and not overwrite:
      return self.index(word, as_tuple=True)
    else:
      indices = tuple(d.add_symbol(w, n=n, overwrite=overwrite) for d, w in zip(self.dicts, word))
      return self.compute_index(indices)

  def add_special_symbol(self, word, n=1, overwrite=False, as_tuple=False):
    if not as_tuple:
      word = (word,)*len(self.dicts)
    self.counts[word] += n
    if word in self.special_symbols and not overwrite:
      return self.index(word, as_tuple=True)
    else:
      index = len(self.special_symbols)
      self.special_symbols.append(word)
      return index

  def update(self, new_dict):
    assert hasattr(new_dict, 'dicts') and hasattr(new_dict, 'counts')
    self.counts.update(new_dict.counts)
    for d1, d2 in zip(self.dicts, new_dict.dicts):
      d1.update(d2)

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
    # same implementation as for a normal Dictionary, except we are adding special symbols
    if padding_factor > 1:
      i = 0
      while len(self) % padding_factor != 0:
        symbol = "madeupword{:04d}".format(i)
        self.add_special_symbol(symbol, n=0, as_tuple=False)
        self.nspecial += 1
        i += 1

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
      nspecial, *factors = [int(f) for f in header.split()[2:]] # header.remove_prefix("# factors ").split()
      factors = tuple(factors)
      header = f.readline().strip()
      assert header.startswith("# sep ")
      sep = header.split()[-1] # header.remove_prefix("# sep ")
      assert nspecial + len(factors) > 0
      lines = f.readlines()
      special_symbols, lines = lines[:nspecial], lines[nspecial:]
      assert len(lines) == sum(factors)
      factor_dicts = [
        io.StringIO(''.join(lines[sum(factors[:i]):sum(factors[:i+1])]))
        for i in range(len(factors))
      ]
    except (AssertionError, ValueError):
      raise ValueError(
        "Incorrect dictionary format, expected"
        "'# factor [special_symbol_count] [factors]' and "
        "'# sep [sep]',"
        "followed by that many lines for special symbols and each factor dictionary."
      )

    dicts = [FactorDictionary.load(factor_dict) for factor_dict in factor_dicts]
    d = cls(sep, dicts=dicts)

    # lifting add_from_file logic from Dictionary to here for special symbols
    for line in special_symbols:
      try:
        line, field = line.rstrip().rsplit(" ", 1)
        if field == "#fairseq:overwrite":
          overwrite = True
          line, field = line.rstrip().rsplit(" ", 1)
        else:
          overwrite = False
        count = int(field)
        word = line
        if (word,)*len(self.dicts) in self.special_symbols and not overwrite:
          raise RuntimeError(
            "Duplicate special word found when loading Dictionary: '{}'. "
            "Duplicate words can overwrite earlier ones by adding the "
            "#fairseq:overwrite flag at the end of the corresponding row "
            "in the dictionary file. If using the Camembert model, please "
            "download an updated copy of the model file.".format(word)
          )
        d.add_special_symbol(word, n=count, overwrite=overwrite, as_tuple=False)
      except ValueError:
        raise ValueError(
          "Incorrect dictionary format, expected"
          "'# factor [special_symbol_count] [factors]' and "
          "'# sep [sep]',"
          "followed by that many lines for special symbols and each factor dictionary."
        )

      return d

  def save(self, f):
    """Stores dictionary into a text file
        a header line lists the factor dict sizes
        then factor dicts are concatenated in order
    """
    if isinstance(f, str):
      PathManager.mkdirs(os.path.dirname(f))
      with PathManager.open(f, "w", encoding="utf-8") as fd:
        return self.save(fd)

    # print a header that will throw an error if we try to load as a normal dict
    header = "# factors " + str(self.nspecial - 3) + " " + " ".join(str(len(d) - d.nspecial) for d in self.dicts)
    print(header, file=f)
    header = "# sep " + self.sep
    print(header, file=f)

    for s in self.special_symbols[3:]:
      print("{} 0".format(s), file=f)
    for d in self.dicts:
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
