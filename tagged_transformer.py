import torch
import torch.nn as nn
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
  Embedding,
  TransformerModel,
  TransformerDecoder,
  base_architecture as transformer_base_architecture
)
from fairseq.modules import AdaptiveSoftmax

@register_model('tagged_transformer')
class TaggedTransformerModel(TransformerModel):
  """
  Tagged Transformer architecture, with:
    - combined embeddings for source and target sequence token-level tags
    - simultaneous prediction of target sequence tokens and tags

  Args:
    encoder (TransformerEncoder): the encoder
    decoder (TransformerDecoder): the decoder

  The Tagged Transformer model provides the following named architectures and
  command-line arguments:

  .. argparse::
    :ref: fairseq.models.tagged_transformer_parser
    :prog:
  """

  # def __init__(self, args, encoder, decoder):
  #   super().__init__(args, encoder, decoder)
  #   self.args = args
  #   self.supports_align_args = True

  @staticmethod
  def add_args(parser):
    TransformerModel.add_args(parser)
    # parser. # ??? any additional arguments needed ? for instance to specify dictionaries

  @classmethod
  def build_embedding(cls, args, dictionary, embed_dim, path=None):
    num_embeddings = dictionary.nspecial + sum(dictionary.factors)
    padding_idx = dictionary.pad()

    emb = SumEmbedding(dictionary, num_embeddings, embed_dim, padding_idx)
    # if provided, load from preloaded dictionaries
    if path:
      embed_dict = utils.parse_embedding(path)
      utils.load_embedding(embed_dict, dictionary, emb) # ???
    return emb

  @classmethod
  def build_decoder(cls, args, tgt_dict, embed_tokens):
    return TaggedTransformerDecoder(
      args,
      tgt_dict,
      embed_tokens,
      no_encoder_attn=getattr(args, "no_cross_attention", False),
    )

class TaggedTransformerDecoder(TransformerDecoder):
  """
  Tagged Transformer decoder consisting of *args.decoder_layers* layers. Each layer
  is a :class:`TransformerDecoderLayer`.

  Args:
    args (argparse.Namespace): parsed command-line arguments
    dictionaries (~List[fairseq.data.Dictionary]): decoding dictionaries
    embed_tokens (torch.nn.Embedding): target side embedding
    no_encoder_attn (bool, optional): whether to attend to encoder outputs
        (default: False).
  """
  def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
    super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
    self.output_projection_indices = nn.Parameter(
      dictionary.factor_indices(torch.arange(len(dictionary)), for_embedding=True),
      requires_grad=False
    ) # this paramater should not be optimized, it's a fixed mapping
    if args.adaptive_softmax_cutoff is not None:
      self.adaptive_softmax = [self.adaptive_softmax] + [
        AdaptiveSoftmax(
          len(dictionary),
          self.output_embed_dim,
          utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
          dropout=args.adaptive_softmax_dropout,
          adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
          factor=args.adaptive_softmax_factor,
          tie_proj=args.tie_adaptive_proj,
        ) for dictionary in dictionaries[1:]
      ] # ??? this will need to function like a single adaptive_softmax
      # but somehow combining results in an outer product fashion the results from multiple
    elif self.share_input_output_embed:
      self.output_projection = nn.Linear(
        self.embed_tokens.weight.shape[1],
        self.embed_tokens.weight.shape[0],
        bias=False,
      )
      self.output_projection.weight = self.embed_tokens.weight
    else:
      # probably better compute all than reuse the first, which may change with fairseq versions...
      self.output_projection = [self.output_projection] + [
        nn.Linear(
          self.output_embed_dim, len(dictionary), bias=False
        ) for dictionary in dictionaries
      ]
      for output_projection in self.output_projection:
        nn.init.normal_(
          output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
        )

  def output_layer(self, features):
    """Project features to the vocabulary size."""
    """
    this couldn't be much easier - factorization of the product space indices
    are already computed in self.output_projection_indices - just need to
    use these as the input to an embedding with the output vector elements
    as weights - just gotta get the dimensions right
    """
    if self.adaptive_softmax is None:
      # self.output_projection(features) is batch indices x sum space
      # need to push sum space through to first axis, so it's the embedding count
      # and also need to collect batch indices into a single axis to form the embedding dim
      # (nn.functional.embedding backprop doesn't support multiple embedding dims)
      sum_projection = self.output_projection(features).T
      sum_vocab_size = sum_projection.shape[0]
      pair_vocab_size = self.output_projection_indices.shape[0]
      pair_projection = add_embeddings_special(
        self.output_projection_indices,
        lambda x: nn.functional.embedding(x,
          sum_projection.reshape(sum_vocab_size, -1))
      ).reshape(pair_vocab_size, -1).T
      return pair_projection
      # first two axes remain batch indices, last axis is pair space projection values
    else:
      return features


class SumEmbedding(nn.Embedding):
  def __init__(self, dictionary, num_embeddings, embedding_dim, padding_idx):
    super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
    self.dictionary = dictionary
    self.weight = Embedding(num_embeddings, embedding_dim, padding_idx).weight

  def forward(self, input):
    input_factors = self.dictionary.factor_indices(input, for_embedding=True)
    return add_embeddings_special(input_factors, super().forward)


def add_embeddings_special(indices, embedding):
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


@register_model_architecture("tagged_transformer", "tagged_transformer")
def base_architecture(args):
  transformer_base_architecture(args)
