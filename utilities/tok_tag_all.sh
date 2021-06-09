#!/bin/bash

INPUT=${1%%+(/)}
SRC=$2
TGT=$3
TOK=$4
VOCAB_SIZE=$5
TAG=$6
TAGGER=$7

MODEL_TYPE=$TOK
if [ "$TOK" == "sp" ]; then MODEL_TYPE=unigram; fi

OUTDIR="$INPUT.$TOK$VOCAB_SIZE/"

python tagging/train_tokenizer.py --vocab-size $VOCAB_SIZE --model-prefix "$OUTDIR/tokens" "\"$INPUT/train.$SRC\",\"$INPUT/train.$TGT\""

for split in train valid test; do
  for lang in $SRC $TGT; do
    python tagging/tok_tag.py --tokenizer "$OUTDIR/tokens.model" --print < "$INPUT/$split.$lang" > "$OUTDIR/$TAG.notags.$lang"
    python tagging/tok_tag.py --tokenizer "$OUTDIR/tokens.model" --tagger "$TAGGER" --print --stats < "$INPUT/$split.$lang" > "$OUTDIR/$TAG.$lang"
  done
  cp $OUTDIR/$TAG.$SRC $OUTDIR/$TAG.srconly.$SRC
  cp $OUTDIR/$TAG.notags.$TGT $OUTDIR/$TAG.srconly.$TGT
  cp $OUTDIR/$TAG.notags.$SRC $OUTDIR/$TAG.tgtonly.$SRC
  cp $OUTDIR/$TAG.$TGT $OUTDIR/$TAG.tgtonly.$TGT
done

