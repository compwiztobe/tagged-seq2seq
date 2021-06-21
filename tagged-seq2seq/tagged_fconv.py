from torch import nn
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fconv import (
  Embedding,
  Linear,
  FConvModel,
  FConvEncoder,
  FConvDecoder,
  base_architecture as fconv_base_architecture
)

from .tagged_model import TaggedModel, TaggedDecoder

@register_model('tagged_fconv')
class TaggedFConvModel(TaggedModel, FConvModel):
  embedding = Embedding

  @staticmethod
  def add_args(parser):
    FConvModel.add_args(parser)
    parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                        help='share decoder input and output embeddings')

  # largely copied from fairseq/models/fconv.py because this functionality
  # is not easily inheritable with the right modifications otherwise
  @classmethod
  def build_model(cls, args, task):
    """Build a new model instance."""
    # make sure that all args are properly defaulted (in case there are any new ones)
    base_architecture(args)

    encoder_embed_dict = None
    if args.encoder_embed_path:
      encoder_embed_dict = utils.parse_embedding(args.encoder_embed_path)
      utils.print_embed_overlap(encoder_embed_dict, task.source_dictionary)

    decoder_embed_dict = None
    if args.decoder_embed_path:
      decoder_embed_dict = utils.parse_embedding(args.decoder_embed_path)
      utils.print_embed_overlap(decoder_embed_dict, task.target_dictionary)

    encoder = TaggedFConvEncoder(
      args,
      dictionary=task.source_dictionary,
      embed_dim=args.encoder_embed_dim,
      embed_dict=encoder_embed_dict,
      convolutions=eval(args.encoder_layers),
      dropout=args.dropout,
      max_positions=args.max_source_positions,
    )
    decoder = TaggedFConvDecoder(
      args,
      dictionary=task.target_dictionary,
      embed_dim=args.decoder_embed_dim,
      embed_dict=decoder_embed_dict,
      convolutions=eval(args.decoder_layers),
      out_embed_dim=args.decoder_out_embed_dim,
      attention=eval(args.decoder_attention),
      dropout=args.dropout,
      max_positions=args.max_target_positions,
      share_embed=args.share_input_output_embed,
    )
    return TaggedFConvModel(encoder, decoder)


class TaggedFConvEncoder(FConvEncoder):
  def __init__(self, argparse_args, dictionary, embed_dim, *args, **kwargs):
    super().__init__(dictionary, embed_dim, *args, **kwargs)

    num_embeddings = dictionary.nspecial + sum(dictionary.factors)
    self.embed_tokens = TaggedFConvModel.build_embedding(argparse_args, dictionary, embed_dim, path=argparse_args.encoder_embed_path)


class TaggedFConvDecoder(TaggedDecoder, FConvDecoder):
  def __init__(self, argparse_args, dictionary, embed_dim, *args, **kwargs):
    super().__init__(dictionary, embed_dim, *args, **kwargs)

    num_embeddings = dictionary.nspecial + sum(dictionary.factors)
    self.embed_tokens = TaggedFConvModel.build_embedding(argparse_args, dictionary, embed_dim, path=argparse_args.decoder_embed_path)

    argparse_args.adaptive_softmax_cutoff = None
    self.share_input_output_embed = argparse_args.share_input_output_embed
    super().init_tagged_output(argparse_args, dictionary, self.embed_tokens)

    self.fc3 = self.output_projection
    self.output_projection = nn.Identity()

  def projection(self, args, dictionary):
    return Linear(args.decoder_out_embed_dim, dictionary.nspecial + sum(dictionary.factors), dropout=args.dropout)

  def forward(self, *args, **kwargs):
    x, extra = super().forward(*args, **kwargs)
    return self.output_layer(x), extra


@register_model_architecture("tagged_fconv", "tagged_fconv")
def base_architecture(args):
  args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
  fconv_base_architecture(args)
  args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
  args.share_input_output_embed = args.share_decoder_input_output_embed
