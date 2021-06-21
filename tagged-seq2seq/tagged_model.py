import torch
from torch import nn

class TaggedModel(): # mixin to a class implementing FairseqEncoderDecoderModel
  @classmethod
  def build_embedding(cls, args, dictionary, embed_dim, path=None):
    num_embeddings = dictionary.nspecial + sum(dictionary.factors)
    padding_idx = dictionary.pad()

    emb = SumEmbedding(cls.embedding, dictionary, num_embeddings, embed_dim, padding_idx)
    # if provided, load from preloaded dictionaries
    if path:
      embed_dict = utils.parse_embedding(path)
      factor_vocab = dictionary.special_dict.keys() + [key for d in dictionary.dicts for key in d]
      utils.load_embedding(embed_dict, factor_vocab, emb)
    return emb


class TaggedDecoder(): # mixin to a class implementing FairseqIncrementalDecoder
  def init_tagged_output(self, args, dictionary, embed_tokens):
    self.output_projection_indices = nn.Parameter(
      dictionary.factor_indices(torch.arange(len(dictionary)), for_embedding=True),
      requires_grad=False
    ) # this paramater should not be optimized, it's a fixed mapping
    if args.adaptive_softmax_cutoff is not None:
      raise NotImplementedError
    elif self.share_input_output_embed:
      self.output_projection = nn.Linear(
        self.embed_tokens.weight.shape[1],
        self.embed_tokens.weight.shape[0],
        bias=False,
      )
      self.output_projection.weight = self.embed_tokens.weight
    else:
      self.output_projection = self.projection(args, dictionary)

  def output_layer(self, features, factored=False):
    """Project features to the vocabulary size."""
    if self.adaptive_softmax is None:
      sum_projection = self.output_projection(features)
      if not factored:
        return self._to_product_space(sum_projection)
      else:
        # return individual output embedding projection for special symbols and each factor
        # without adding them together
        sizes = [self.dictionary.nspecial, *self.dictionary.factors]
        bounds = [(sum(sizes[:i]), sum(sizes[:i+1])) for i in range(len(sizes))]
        return (
          sum_projection,
          *[sum_projection[:,:,start:end] for start, end in bounds]
        )
    else:
      return features

  def _to_product_space(self, sum_projection):
    """
    this couldn't be much easier - factorization of the product space indices
    are already computed in self.output_projection_indices - just need to
    use these as the input to an embedding with the output vector elements
    as weights - just gotta get the dimensions right
    """
    # sum_projection = self.output_projection(features) is batch indices x sum_space
    # need to push sum space through to first axis, so it's the embedding count
    # and also need to collect batch indices into a single axis to form the embedding dim
    # (nn.functional.embedding backprop doesn't support multiple embedding dims)
    sum_projection = sum_projection.T
    vocab_size = sum_projection.shape[0]
    batch_size = sum_projection.shape[1:]
    pair_projection = add_embeddings(
      self.output_projection_indices,
      lambda x: nn.functional.embedding(x,
        sum_projection.reshape(vocab_size, -1))
    ).reshape(-1, *batch_size).T
    return pair_projection
    # first two axes remain batch indices, last axis is pair space projection values


class SumEmbedding(nn.Embedding):
  def __init__(self, embedding, dictionary, num_embeddings, embedding_dim, padding_idx):
    super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
    self.dictionary = dictionary
    self.weight = embedding(num_embeddings, embedding_dim, padding_idx).weight

  def forward(self, input):
    input_factors = self.dictionary.factor_indices(input, for_embedding=True)
    return add_embeddings(input_factors, super().forward)


def add_embeddings(indices, embedding):
  # indices is any size, output is indices.shape[:-1] x embedding.shape[1:]
  # summed along last axis of indices (the factor axis)
  # indices should be between 0 and embedding.shape[0] - 1
  # some indices may be -1 indicating a special symbol
  # for which we only sum over one vocab position
  special_indices = indices < 0
  indices[special_indices] = 0
  embeddings = embedding(indices)
  embeddings[special_indices,:] = 0
  return embeddings.sum(axis=indices.dim() - 1)
