import math

from fairseq import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

@register_criterion("cross_entropy_decomposition")
class CrossEntropyDecompositionCriterion(LabelSmoothedCrossEntropyCriterion):
  def forward(self, model, sample, reduce=True):
    # splitting this into two steps here to be able to reuse the sum space
    # projection computation (the bulk of the decoder) while having access
    # to both that and the subsequent pair space projection
    output_layer = model.decoder.output_layer
    model.decoder.output_layer = lambda *args: output_layer(*args, factored=True)
    (net_output, special_output, *factor_outputs), _ = model(**sample["net_input"])
    model.decoder.output_layer = output_layer
    net_output = (model.decoder._to_pair_space(net_output), _)
    # normally this is done in one pass by model.forward with factored=False (default)

    # copied from super().forward because it can't be separated otherwise
    loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
    sample_size = (
      sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
    )
    logging_output = {
      "loss": loss.data,
      "nll_loss": nll_loss.data,
      "ntokens": sample["ntokens"],
      "nsentences": sample["target"].size(0),
      "sample_size": sample_size,
    }
    if self.report_accuracy:
      n_correct, total = self.compute_accuracy(model, net_output, sample)
      logging_output["n_correct"] = utils.item(n_correct.data)
      logging_output["total"] = utils.item(total.data)

    #####

    # computing additional component loss terms

    sample = sample["target"].reshape(-1) # sentence dimension unimportant and we need
    # to be able to mask special symbols at arbitrary positions without changing shape
    special_output = special_output.reshape(-1, special_output.size(-1))
    factor_outputs = [factor_output.reshape(-1, factor_output.size(-1)) for factor_output in factor_outputs]
    # this breaks --ignore-prefix-size unless we can recover lost sentence positions and modify get_lprobs_and_target accordingly

    factored_sample = model.decoder.dictionary.factor_indices(sample, for_embedding=False)

    special_indices = sample < model.decoder.dictionary.nspecial

    special_output = (special_output[special_indices], _)
    factor_outputs = [(factor_output[~special_indices], _) for factor_output in factor_outputs]
    special_sample = factored_sample[special_indices, 0]
    factor_samples = [factored_sample[~special_indices, i] for i in range(len(model.decoder.dictionary.factors))]

    special_sample = {"target": special_sample}
    factor_samples = [{"target": factor_sample} for factor_sample in factor_samples]

    loss_special, nll_loss_special = self.compute_loss(model, special_output, special_sample, reduce=reduce)
    # _, nll_loss_special = ...

    loss_factors, nll_loss_factors = zip(*[
      self.compute_loss(model, factor_output, factor_sample, reduce=reduce)
      for factor_output, factor_sample in zip(factor_outputs, factor_samples)
    ])
    # although we don't really use the label smoothed loss_factors, so could drop those with
    # nll_loss_factors = [self.compute_loss(...)[1] for ...]

    logging_output.update({
      "nll_special": nll_loss_special.data,
      # "ntokens_special": ,
      # "ntokens_nonspecial": ,
    })
    logging_output.update({
      "nll_factor%d" % i: nll_loss_factor.data for i, nll_loss_factor in enumerate(nll_loss_factors)
    })

    return loss, sample_size, logging_output

  @classmethod
  def reduce_metrics(cls, logging_outputs):
    super().reduce_metrics(logging_outputs)

    # this is a classmethod with no knowledge of instances and their factors
    # so we need to infer from the logging outputs
    factor_count = max(
      int(key[len("nll_factor"):])
      for log in logging_outputs
      for key in log.keys()
      if key.startswith("nll_factor")
    ) + 1

    nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)

    nll_special_sum = sum(log.get("nll_special", 0) for log in logging_outputs)

    nll_factor_sums = [
      sum(log.get("nll_factor%d" % i, 0) for log in logging_outputs)
      for i in range(factor_count)
    ]

    ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)

    # mightn't we need special token counts?
    # ntokens_special = sum(log.get("ntokens_special", 0) for log in logging_outputs)
    # ntokens_nonspecial = sum(log.get("ntokens_nonspecial", 0) for log in logging_outputs)

    nll_loss = nll_loss_sum / ntokens / math.log(2)
    nll_special = nll_special_sum / ntokens / math.log(2)
    nll_factors = [nll_factor_sum / ntokens / math.log(2) for nll_factor_sum in nll_factor_sums]
    nll_nonspecial = sum(nll_factors)
    metrics.log_scalar(
      "nll_special", nll_special, ntokens, round=6
    )
    metrics.log_scalar(
      "nll_nonspecial", nll_nonspecial, ntokens, round=3
    )
    for i, nll_factor in enumerate(nll_factors):
      metrics.log_scalar(
        "nll_factor%d" % i, nll_factor, ntokens, round=3
      )

    metrics.log_scalar(
      "nll_renorm", nll_loss - nll_special - nll_nonspecial, round = 3
    )
