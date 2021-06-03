#!/bin/bash

OUTDIR=$1
SRC=$3
TGT=$4
TOK=$5
SIZE=$6

python utilities/train_tokenizer.py $OUTDIR/notags.train.{$SRC,$TGT} --model-type $TOK --vocab-size $SIZE --model-prefix $OUTDIR/$TOK$SIZE

if [[ "$TAG" == *"ner"* ]]; then
  NER_STATS="--ner-stats"
else
  NER_STATS=
fi

for dataset in notags $TAG.srconly $TAG.tgtonly $TAG.; do
  for split in train valid test test.noNE test.someNE; do
    for lang in $SRC $TGT; do
      python utilities/tokenize.py -m $OUTDIR/$TOK$SIZE.model $NER_STATS --tag-types=$TAG --print < $OUTDIR/$dataset.$split.$lang > $OUTDIR/$TOK$SIZE.$dataset.$split.$lang
    done
  done
done
