#!/bin/bash

OUTDIR=$1
SRC=$2
TGT=$3
TOK=$4
SIZE=$5

python utilities/train_tokenizer.py $OUTDIR/notags.train.{$SRC,$TGT} --model-type $TOK --vocab-size ${SIZE//k/000} --model-prefix $OUTDIR/$TOK$SIZE

tokenize() {
  dataset="$1"
  tagtypes="$2"
  args="$3"
  python utilities/tokenize.py -m $OUTDIR/$TOK$SIZE.model --tag-types $tagtypes $args < "$OUTDIR/$dataset" > "$OUTDIR/$TOK$SIZE.$dataset"
}

for lang in $SRC $TGT; do
  for tags in notags ner.srconly ner.tgtonly ner; do
    for split in train valid test test.someNE test.noNE; do
      printf '%s' "$tags.$split.$lang: "
      tokenize $tags.$split.$lang ner --ner-stats
    done
  done
  for tags in upos.srconly upos.tgtonly upos; do
    for split in train valid test; do
      tokenize $tags.$split.$lang upos
    done
  done
done
