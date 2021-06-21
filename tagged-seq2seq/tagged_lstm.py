from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.lstm import (
  Embedding,
  Linear,
  LSTMModel,
  LSTMEncoder,
  LSTMDecoder,
  DEFAULT_MAX_SOURCE_POSITIONS,
  DEFAULT_MAX_TARGET_POSITIONS,
  base_architecture as lstm_base_architecture
)

from .tagged_model import TaggedModel, TaggedDecoder

@register_model('tagged_lstm')
class TaggedLSTMModel(TaggedModel, LSTMModel):
  embedding = Embedding

  # largely copied from fairseq/models/lstm.py because this functionality
  # is not easily inheritable with the right modifications otherwise
  @classmethod
  def build_model(cls, args, task):
    """Build a new model instance."""
    # make sure that all args are properly defaulted (in case there are any new ones)
    lstm_base_architecture(args)

    if args.encoder_layers != args.decoder_layers:
      raise ValueError("--encoder-layers must match --decoder-layers")

    max_source_positions = getattr(
      args, "max_source_positions", DEFAULT_MAX_SOURCE_POSITIONS
    )
    max_target_positions = getattr(
      args, "max_target_positions", DEFAULT_MAX_TARGET_POSITIONS
    )

    pretrained_encoder_embed = cls.build_embedding(args, task.source_dictionary, args.encoder_embed_dim)

    if args.share_all_embeddings:
      # double check all parameters combinations are valid
      if task.source_dictionary != task.target_dictionary:
        raise ValueError("--share-all-embeddings requires a joint dictionary")
      if args.decoder_embed_path and (
        args.decoder_embed_path != args.encoder_embed_path
      ):
        raise ValueError(
          "--share-all-embed not compatible with --decoder-embed-path"
        )
      if args.encoder_embed_dim != args.decoder_embed_dim:
        raise ValueError(
          "--share-all-embeddings requires --encoder-embed-dim to "
          "match --decoder-embed-dim"
        )
      pretrained_decoder_embed = pretrained_encoder_embed
      args.share_decoder_input_output_embed = True
    else:
      # separate decoder input embeddings
      pretrained_decoder_embed = cls.build_embedding(args, task.target_dictionary, args.encoder_embed_dim)

    # one last double check of parameter combinations
    if args.share_decoder_input_output_embed and (
      args.decoder_embed_dim != args.decoder_out_embed_dim
    ):
      raise ValueError(
        "--share-decoder-input-output-embeddings requires "
        "--decoder-embed-dim to match --decoder-out-embed-dim"
      )

    if args.encoder_freeze_embed:
      pretrained_encoder_embed.weight.requires_grad = False
    if args.decoder_freeze_embed:
      pretrained_decoder_embed.weight.requires_grad = False

    encoder = LSTMEncoder(
      dictionary=task.source_dictionary,
      embed_dim=args.encoder_embed_dim,
      hidden_size=args.encoder_hidden_size,
      num_layers=args.encoder_layers,
      dropout_in=args.encoder_dropout_in,
      dropout_out=args.encoder_dropout_out,
      bidirectional=args.encoder_bidirectional,
      pretrained_embed=pretrained_encoder_embed,
      max_source_positions=max_source_positions,
    )
    decoder = TaggedLSTMDecoder(
      args,
      dictionary=task.target_dictionary,
      embed_dim=args.decoder_embed_dim,
      hidden_size=args.decoder_hidden_size,
      out_embed_dim=args.decoder_out_embed_dim,
      num_layers=args.decoder_layers,
      dropout_in=args.decoder_dropout_in,
      dropout_out=args.decoder_dropout_out,
      attention=utils.eval_bool(args.decoder_attention),
      encoder_output_units=encoder.output_units,
      pretrained_embed=pretrained_decoder_embed,
      share_input_output_embed=args.share_decoder_input_output_embed,
      adaptive_softmax_cutoff=(
        utils.eval_str_list(args.adaptive_softmax_cutoff, type=int)
        if args.criterion == "adaptive_loss"
        else None
      ),
      max_target_positions=max_target_positions,
      residuals=False,
    )
    return cls(encoder, decoder)


class TaggedLSTMDecoder(TaggedDecoder, LSTMDecoder):
  def __init__(self, argparse_args, dictionary, *args, **kwargs):
    super().__init__(dictionary, *args, **kwargs)
    super().init_tagged_output(argparse_args, dictionary, self.embed_tokens)

  def projection(self, args, dictionary):
    return Linear(args.decoder_out_embed_dim, dictionary.nspecial + sum(dictionary.factors), bias=False)


@register_model_architecture("tagged_lstm", "tagged_lstm")
def base_architecture(args):
  args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
  lstm_base_architecture(args)
