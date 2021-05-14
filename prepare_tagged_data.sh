#!/bin/bash

PREFIX=$1
TAG=$2
SRC=$3
TGT=$4
TOK=$5
SIZE=$6

# this is a template more than anything, it's likely we should run each of these steps separately

# repair some unicode mismatch
# already done - don't trample! (or waste clock cycles)
#for lang in $SRC $TGT; do
#  python utilities/strip_tags.py "<&&&>" < $PREFIX.$TAG.$lang > $PREFIX.$lang
#done

splits="train,3000000 valid,100000 test,100000"

# build notags split
utilities/generate_splits.sh $PREFIX 3m $TAG $SRC $TGT "$splits" "--dataset-size $(wc -l $PREFIX.$SRC | cut -d" " -f1)"

# copy those split indices to srconly, tgtonly, tagged
utilities/copy_splits.sh $PREFIX 3m $TAG $SRC $TGT "$splits"

# if ner, gen some/noNE splits
if [[ "$TAG" == *"ner"* ]]; then
  utilities/ne_splits.sh $PREFIX 3m $TAG $SRC $TGT
fi

# now ready to train sentence piece

utilities/tokenize.sh $PREFIX.3m $TAG $SRC $TGT $TOK $SIZE
