#!/bin/bash

PREFIX=$1
OUTSUFFIX=$2
TAG=$3
SRC=$4
TGT=$5
SPLITS="$6"
DATASIZEORREUSE="$7"

OUTDIR=$PREFIX.$OUTSUFFIX

mkdir $OUTDIR

paste -d '\n' $PREFIX.$TAG.{$SRC,$TGT} | python utilities/random_splits.py $OUTDIR/$TAG. $SPLITS $DATASIZEORREUSE --skip 2 || exit 1
# quit this script if it failed to write the splits, before we mangle any existing splits with the next command
# less worried about doing that for the later copied splits but treating this split as the reference, don't want to corrupt

for split in $SPLITS; do
  utilities/split_even_odd.sh $OUTDIR/$TAG.${split%,*} $SRC $TGT
done
