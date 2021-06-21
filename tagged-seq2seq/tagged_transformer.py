from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
  Embedding,
  TransformerModel,
  TransformerDecoder,
  base_architecture as transformer_base_architecture
)

from .tagged_model import TaggedModel, TaggedDecoder

@register_model('tagged_transformer')
class TaggedTransformerModel(TaggedModel, TransformerModel):
  embedding = Embedding

  @classmethod
  def build_decoder(cls, args, tgt_dict, embed_tokens):
    return TaggedTransformerDecoder(
      args,
      tgt_dict,
      embed_tokens,
      no_encoder_attn=getattr(args, "no_cross_attention", False),
    )


class TaggedTransformerDecoder(TaggedDecoder, TransformerDecoder):
  def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
    super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
    super().init_tagged_output(args, dictionary, embed_tokens)

  def projection(self, args, dictionary):
    output_layer = nn.Linear(self.output_embed_dim, dictionary.nspecial + sum(dictionary.factors), bias=False)
    nn.init.normal_(output_layer.weight, mean=0, std=self.output_embed_dim ** -0.5)
    return output_layer


@register_model_architecture("tagged_transformer", "tagged_transformer")
def base_architecture(args):
  transformer_base_architecture(args)
