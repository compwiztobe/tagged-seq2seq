from math import prod # as product # requires python 3.8
from itertools import product # as direct_product
from collections import Counter
from torch import is_tensor
from fairseq.data import Dictionary

class TupleDictionary(Dictionary):
  """
  convert multiple dictionaries with their own indices
  into a single dictionary with factored indices
  """
  def __init__(self, sep, *dicts):
    self.sep = sep
    self.dicts = dicts
    self.counts = Counter()
    self.bos_word, self.unk_word, self.pad_word, self.eos_word = [
      tuple(getattr(d, attr) for d in dicts)
      for attr in ['bos_word', 'unk_word', 'pad_word', 'eos_word']
    ]
    self.bos_index, self.unk_index, self.pad_index, self.eos_index = [
      self.compute_index(getattr(d, attr) for d in dicts)
      for attr in ['bos_index', 'unk_index', 'pad_index', 'eos_index']
    ]
    # self.extra_special_symbols = product(d.extra_special_symbols for d in dicts)
    self.nspecial = prod(d.nspecial for d in dicts)

  def __getattr__(self, attr):
    try:
      return tuple(getattr(d, attr) for d in self.dicts)
    except AttributeError:
      raise AttributeError("'%s' object has no attribute '%s'", (type(self), attr))

  @property
  def factors(self):
    return tuple(len(d) for d in self.dicts)

  @property
  def symbols(self):
    return list(product(*[d.symbols for d in self.dicts]))

  @property
  def count(self):
    return [self.counts[sym] for sym in symbols]

  @property
  def indices(self):
    return {
      symbols: self.compute_index(d.indices[symbol] for d, symbol in zip(self.dicts, symbols))
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
    return hasattr(other, 'dicts') and self.dicts == other.dicts

  def __len__(self):
    return prod(self.factors())

  def __getitem__(self, index):
    indices = self.factor_index(index)
    return tuple(d.symbols[i] for d, i in zip(self.dicts, indices))

  def __contains__(self, syms):
    return all(sym in d.symbols for sym, d in zip(syms, self.dicts))

  def index(self, syms, as_tuple=False):
    """Returns the index of the specified symbol"""
    assert isinstance(syms, str) or as_tuple
    if not as_tuple:
      syms = tuple(syms.split(self.sep))
    assert isinstance(syms, tuple) and len(syms) == len(self.dicts)
    return self.compute_index(d.index(sym) for d, sym in zip(self.dicts, syms))

  def unk_string(self, escape=False, as_tuple=False):
    unk = tuple(d.unk_string(escape) for d in self.dicts)
    if as_tuple:
      return unk
    else:
      return self.sep.join(unk)

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
      bpe_symbols = [None for _ in self.dicts]
    if escape_unks is False:
      escape_unks = [False for _ in self.dicts]
    if extra_symbols_to_ignore is None:
      extra_symbols_to_ignore = [None for _ in self.dicts]
    if unk_strings is None:
      unk_strings = [None for _ in self.dicts]

    if not factored_indices:
      tensor = [self.factor_index(i) for i in tensor]
    print(tensor)
    print(list(zip(*tensor)))
    strings = [
      [
        d.string([index], bpe_symbol, escape_unk, extra_symbols, unk_string)
        for index in indices
      ]
      for d, indices, bpe_symbol, escape_unk, extra_symbols, unk_string
      in zip(self.dicts, zip(*tensor), bpe_symbols, escape_unks, extra_symbols_to_ignore, unk_strings)
    ]
    return " ".join(separator.join(t) for t in zip(*strings))

  def add_symbol(self, word, n=1, overwrite=False, as_tuple=False):
    if not as_tuple:
      word = tuple(word.split(self.sep))
    self.counts.update(Counter({word: n}))
    indices = tuple(d.add_symbol(w, n=n, overwrite=overwrite) for d, w in zip(self.dicts, word))
    return self.compute_index(indices)

  def update(self, new_dict):
    assert hasattr(new_dict, 'dicts') and hasattr(new_dict, 'counts')
    for d1, d2 in zip(self.dicts, new_dict.dicts):
      d1.update(d2)
    self.counts.update(new_dict.counts)

  def finalize(self, threshold=-1, nwords=-1, padding_factor=[8]):
    # a couple ways to finalize:
    # threshold, nwords, and padding_factor can be applied at the tuple level
    # though this may complicate how it propagates to the factor dictionaries
    # and keeps this factorization valid
    # or, threshold, nwords, and padding_factor can be applied to each factor
    # but then we probably want to apply differently for each, or apply to only one
    # threshold only the token vocab (use full tag vocab)
    # nwords limits only the size of the subword token vocab (again, use full tag vocab)
    # padding_factor only applies to token vocab, likely to be the largest factor and
    # therefore guarantees the tuple vocab has the same factor, while minimizing the extra increase in size
    if isinstance(threshold, int):
      threshold = [threshold for _ in self.dicts]
    if isinstance(nwords, int):
      nwords = [nwords for _ in self.dicts]
    if isinstance(padding_factor, int):
      padding_factor = [padding_factor for _ in self.dicts]
    if len(threshold) < len(self.dicts):
      threshold = threshold + [-1] * (len(self.dicts) - len(threshold))
    if len(nwords) < len(self.dicts):
      nwords = nwords + [-1] * (len(self.dicts) - len(nwords))
    if len(padding_factor) < len(self.dicts):
      padding_factor = padding_factor + [1] * (len(self.dicts) - len(padding_factor))
    for d, t, n, p in zip(self.dicts, threshold, nwords, padding_factor):
      d.finalize(t, n, p)
    # this will mess with the tuple level counts

  def pad_to_multiple_(self, padding_factor):
    raise NotImplementedError

  # need this inherited from Dictionary, since LegacyFairseqTask calls it on the task's
  # dictionary class (not statically)
  # @classmethod
  # def load(cls, f):
  #   raise NotImplementedError

  def add_from_file(self, f):
    # loop through the file as in inherited implementation,
    # but splitting vocab entries on sep and adding to appropriate factor dictionaries
    # actually this is no different from inherited implementation, and so with count support
    # at tuple level (and add_symbol properly implemented), this would not be needed at all
    if isinstance(f, str):
      try:
        with open(PathManager.get_local_path(f), "r", encoding="utf-8") as fd:
          self.add_from_file(fd)
      except FileNotFoundError as fnfe:
        raise fnfe
      except UnicodeError:
        raise Exception(
          "Incorrect encoding detected in {}, please "
          "rebuild the dataset".format(f)
        )
      return

      lines = f.readlines()
      indices_start_line = self._load_meta(lines)

      for line in lines[indices_start_line:]:
        try:
          line = line.rstrip()
          if line.endswith("#fairseq:overwrite"):
            overwrite = True
            line, _ = line.rsplit(" ", 1)
          else:
            overwrite = False
            word = line
            if word in self and not overwrite:
              raise RuntimeError(
                "Duplicate word found when loading Dictionary: '{}'. "
                "Duplicate words can overwrite earlier ones by adding the "
                "#fairseq:overwrite flag at the end of the corresponding row "
                "in the dictionary file. If using the Camembert model, please "
                "download an updated copy of the model file.".format(word)
                )
              self.add_symbol(word, overwrite=overwrite)
            except ValueError:
              raise ValueError(
                "Incorrect dictionary format, expected '<token> [flags]'"
                )

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
