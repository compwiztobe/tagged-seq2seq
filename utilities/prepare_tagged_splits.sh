#!/bin/bash

PREFIX=$1
SUFFIX=$2
TAG=$3
SRC=$4
TGT=$5
SPLITS=$6

# this is a template more than anything, it's likely we should run each of these steps separately

# repair some unicode mismatch
# already done - don't trample! (or waste clock cycles)
#for lang in $SRC $TGT; do
#  python utilities/strip_tags.py "<&&&>" < $PREFIX.$TAG.$lang > $PREFIX.$lang
#done

# splits="train,3000000 valid,100000 test,100000"
splits="$SPLITS"

if [[ $splits != *--reuse-indices* ]]; then
  dataset_size="--dataset-size $(wc -l $PREFIX.$SRC | cut -d" " -f1)"
else
  dataset_size=""
fi

# build notags split
utilities/generate_splits.sh $PREFIX $SUFFIX $TAG $SRC $TGT "$splits" "$dataset_size"

# copy those split indices to srconly, tgtonly, tagged
utilities/copy_splits.sh $PREFIX $SUFFIX $TAG $SRC $TGT "$splits"

# if ner, gen some/noNE splits
if [[ "$TAG" == *"ner"* ]]; then
  utilities/ne_splits.sh $PREFIX $SUFFIX $TAG $SRC $TGT
fi
