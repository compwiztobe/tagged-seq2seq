#!/bin/bash

PREFIX=$1
OUTSUFFIX=$2
TAG=$3
SRC=$4
TGT=$5
SPLITS="$6"

OUTDIR=$PREFIX.$OUTSUFFIX

REUSE=${7-$OUTDIR/$TAG.}

mkdir $OUTDIR

paste -d '\n' $PREFIX.{$SRC,$TGT} | python utilities/random_splits.py $OUTDIR/notags. $SPLITS --reuse-indices $REUSE --skip 2 &&
for split in $SPLITS; do
  utilities/split_even_odd.sh $OUTDIR/notags.${split%,*} $SRC $TGT
done

paste -d '\n' $PREFIX.{$TAG.$SRC,$TGT} | python utilities/random_splits.py $OUTDIR/$TAG.srconly. $SPLITS --reuse-indices $REUSE --skip 2 &&
for split in $SPLITS; do
  utilities/split_even_odd.sh $OUTDIR/$TAG.srconly.${split%,*} $SRC $TGT
done

paste -d '\n' $PREFIX.{$SRC,$TAG.$TGT} | python utilities/random_splits.py $OUTDIR/$TAG.tgtonly. $SPLITS --reuse-indices $REUSE --skip 2 &&
for split in $SPLITS; do
  utilities/split_even_odd.sh $OUTDIR/$TAG.tgtonly.${split%,*} $SRC $TGT
done
