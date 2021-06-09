#!/bin/bash

PREFIX=$1
OUTSUFFIX=$2
TAG=$3
SRC=$4
TGT=$5
SEP=${6-"<&&&>"}

OUTDIR=$PREFIX.$OUTSUFFIX

# generate some/noNE splits of tagged data
out="$(paste -d '\n' $OUTDIR/$TAG.test.{$SRC,$TGT} | python utilities/ne_split.py $OUTDIR/$TAG.test "$SEP" 2)"
echo "$out"
utilities/split_even_odd.sh $OUTDIR/$TAG.test.noNE $SRC $TGT
utilities/split_even_odd.sh $OUTDIR/$TAG.test.someNE $SRC $TGT
# index the test indices by the some/noNE indices
python utilities/index_file.py $OUTDIR/$TAG.test.idx < $OUTDIR/$TAG.test.noNE.idx > $OUTDIR/$TAG.test.noNE.idx.tmp
python utilities/index_file.py $OUTDIR/$TAG.test.idx < $OUTDIR/$TAG.test.someNE.idx > $OUTDIR/$TAG.test.someNE.idx.tmp
# repair the idx files
mv $OUTDIR/$TAG.test.noNE.idx{.tmp,}
mv $OUTDIR/$TAG.test.someNE.idx{.tmp,}
# get the total lengths these new splits (wc would also work...)
noNE=$(echo "$out" | grep "found with no NEs" | cut -d" " -f1)
someNE=$(echo "$out" | grep "found with NEs" | cut -d" " -f1)
# copy to other datasets
utilities/copy_splits.sh $PREFIX $OUTSUFFIX $TAG $SRC $TGT "test.noNE,$noNE test.someNE,$someNE"
