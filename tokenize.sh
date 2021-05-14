#!/bin/bash

OUTDIR=$1
TAG=$2
SRC=$3
TGT=$4
TOK=$5

python utilities/train_tokenizer.py $OUTDIR/notags.test.$SRC,$PREFIX.3m/notags.test.$TGT --model-type $TOK --vocab-size $SIZE --model-prefix $OUTDIR/$TOK$SIZE

if [[ "$TAG" == *"ner"* ]]; then
  NER_STATS="--ner-stats"
fi

for dataset in notags $TAG.srconly $TAG.tgtonly $TAG.; do
  for split in train valid test test.noNE test.someNE; do
    for lang in $SRC $TGT; do
      python utilities/tokenize.py -m $OUTDIR/$TOK$SIZE $NER_STATS --tag-types=$TAG --print < $OUTDIR/$dataset.$split.$lang > $OUTDIR/$TOK$SIZE.$dataset.$split.$lang
    done
  done
done
