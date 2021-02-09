from math import prod # as product # requires python 3.8
from itertools import product # as direct_product
from torch import is_tensor
from fairseq.data import Dictionary

class TupleDictionary(Dictionary):
  """
  convert multiple dictionaries with their own indices
  into a single dictionary with factored indices
  """
  def __init__(self, *dictionaries):
    self.dictionaries = dictionaries
    self.bos_word, self.unk_word, self.pad_word, self.eos_word = [
      tuple(getattr(dictionary, attr) for dictionary in dictionaries)
      for attr in ['bos_word', 'unk_word', 'pad_word', 'eos_word']
    ]
    self.bos_index, self.unk_index, self.pad_index, self.eos_index = [
      self.compute_index(getattr(dictionary, attr) for dictionary in dictionaries)
      for attr in ['bos_index', 'unk_index', 'pad_index', 'eos_index']
    ]
    # self.extra_special_symbols = product(dictionary.extra_special_symbols for dictionary in dictionaries)
    self.nspecial = prod(dictionary.nspecial for dictionary in dictionaries)

  def __getattr__(self, attr):
    try:
      return tuple(getattr(dictionary, attr) for dictionary in self.dictionaries)
    except AttributeError:
      raise AttributeError("'%s' object has no attribute '%s'", (type(self), attr))

  @property
  def factors(self):
    return tuple(len(dictionary) for dictionary in self.dictionaries)

  @property
  def symbols(self):
    return list(product(*[dictionary.symbols for dictionary in self.dictionaries]))

  @property
  def count(self):
    return list(prod(x) for x in product(*self.__getattr__('count')))

  @property
  def indices(self):
    return {
      symbols: self.compute_index(dictionary.indices[symbol] for dictionary, symbol in zip(self.dictionaries, symbols))
      for symbols in self.symbols
    }

  def factor_index(self, index):
    return tuple(
      index%prod(self.factors[i:])//prod(self.factors[i+1:])
      for i in range(len(self.factors))
    )

  def compute_index(self, indices):
    return sum(
      prod(self.factors[i+1:])*index
      for i, index in enumerate(indices)
    )

  def __eq__(self, other):
    return hasattr(other, 'dictionaries') and self.dictionaries == other.dictionaries

  def __len__(self):
    return prod(self.factors())

  def __getitem__(self, index):
    indices = self.factor_index(index)
    return tuple(dictionary.symbols[i] for dictionary, i in zip(self.dictionaries, indices))

  def __contains__(self, syms):
    return all(sym in dictionary.symbols for sym, dictionary in zip(syms, self.dictionaries))

  def index(self, syms):
    """Returns the index of the specified symbol"""
    assert isinstance(syms, tuple)
    return self.compute_index(dictionary.index(sym) for dictionary, sym in zip(self.dictionaries, syms))

  def unk_string(self, escape=False):
    return tuple(dictionary.unk_string(escape) for dictionary in self.dictionaries)

  def string(
    self,
    tensor,
    factored_indices=False,
    separator=" ",
    bpe_symbols=None,
    escape_unks=False,
    extra_symbols_to_ignore=None,
    unk_strings=None,
    **kwargs
  ):
    # raise NotImplementedError
    if is_tensor(tensor) and tensor.dim() == 2 + factored_indices:
      return "\n".join(
        self.string(t, factored_indices, bpe_symbols, escape_unks, extra_symbols_to_ignore, unk_strings)
        for t in tensor
      )

    if bpe_symbols is None:
      bpe_symbols = [None for _ in self.dictionaries]
    if escape_unks is False:
      escape_unks = [False for _ in self.dictionaries]
    if extra_symbols_to_ignore is None:
      extra_symbols_to_ignore = [None for _ in self.dictionaries]
    if unk_strings is None:
      unk_strings = [None for _ in self.dictionaries]

    if not factored_indices:
      tensor = [self.factor_index(i) for i in tensor]
    print(tensor)
    print(list(zip(*tensor)))
    strings = [
      [
        dictionary.string([index], bpe_symbol, escape_unk, extra_symbols, unk_string)
        for index in indices
      ]
      for dictionary, indices, bpe_symbol, escape_unk, extra_symbols, unk_string
      in zip(self.dictionaries, zip(*tensor), bpe_symbols, escape_unks, extra_symbols_to_ignore, unk_strings)
    ]
    return " ".join(separator.join(t) for t in zip(*strings))

  def add_symbol(self, word, n=1, overwrite=False):
    raise NotImplementedError

  def update(self, new_dict):
    raise NotImplementedError

  def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
    raise NotImplementedError

  def pad_to_multiple_(self, padding_factor):
    raise NotImplementedError

  @classmethod
  def load(cls, f):
    raise NotImplementedError

  def add_from_file(self, f):
    raise NotImplementedError

  @staticmethod
  def _add_file_to_dictionary_single_worker(
    filename, tokenize, eos_word, worker_id=0, num_workers=1
  ):
    raise NotImplementedError

  @staticmethod
  def add_file_to_dictionary(filename, dict, tokenize, num_workers):
    raise NotImplementedError
