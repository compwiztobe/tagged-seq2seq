import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
  TransformerModel,
  TransformerDecoder,
  base_architecture as transformer_base_architecture
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import AdaptiveSoftmax
from torch import Tensor

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

    emb = sumEmbedding(dictionary, num_embeddings, embed_dim, padding_idx)
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

  def forward( # ??? maybe this requires no change, only output_layer
    self,
    prev_output_tokens,
    encoder_out: Optional[EncoderOut] = None,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    features_only: bool = False,
    full_context_alignment: bool = False,
    alignment_layer: Optional[int] = None,
    alignment_heads: Optional[int] = None,
    src_lengths: Optional[Any] = None,
    return_all_hiddens: bool = False,
  ):
    """
    Args:
      prev_output_tokens (LongTensor): previous decoder outputs of shape
        `(batch, tgt_len)`, for teacher forcing
      encoder_out (optional): output from the encoder, used for
        encoder-side attention
      incremental_state (dict): dictionary used for storing state during
        :ref:`Incremental decoding`
      features_only (bool, optional): only return features without
        applying output layer (default: False).
      full_context_alignment (bool, optional): don't apply
        auto-regressive mask to self-attention (default: False).

    Returns:
      tuple:
        - the decoder's output of shape `(batch, tgt_len, vocab)`
        - a dictionary with any model-specific outputs
    """
    x, extra = self.extract_features(
      prev_output_tokens,
      encoder_out=encoder_out,
      incremental_state=incremental_state,
      full_context_alignment=full_context_alignment,
      alignment_layer=alignment_layer,
      alignment_heads=alignment_heads,
    )
    if not features_only:
      x = self.output_layer(x)
    return x, extra

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
      return add_embeddings_special(
        self.output_projection_indices,
        lambda x: nn.functional.embedding(x, self.output_projection(features).T)
      ).T
      # first two axes remain batch indices, last axis is pair space projection values
    else:
      return features


def sumEmbedding(dictionary, num_embeddings, embedding_dim, padding_idx):
  m = SumEmbedding(dictionary, num_embeddings, embedding_dim, padding_idx=padding_idx)
  nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
  nn.init.constant_(m.weight[padding_idx], 0)
  return m

class SumEmbedding(nn.Embedding):
  def __init__(self, dictionary, *args, **kwargs):
    self.dictionary = dictionary
    super().__init__(*args, **kwargs)

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
