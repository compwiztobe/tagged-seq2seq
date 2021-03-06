#!/bin/bash

INPUT_DIR=$1
LANGPAIR=$2
SRC=${LANGPAIR%-*}
TGT=${LANGPAIR#*-}
TOK=$3
TAGABLATION=$4
if [[ $TAGABLATION == *-* ]]; then
  TAG=${TAGABLATION%-*}
  ABLATION=${TAGABLATION#*-}
else
  TAG=$TAGABLATION
  ABLATION=""
fi
OUTPUT_PREFIX=$5

if [[ $6 == "shared" ]]; then
  SHARED=shared
else
  SHARED=
fi
SUBSPLIT=$7

ADDITIONAL_OPTIONS=$8

TAG_SEP=${TAG_SEP:-"<&&&>"}

### Python Virtual Env ###
echo
echo Start python ...
source ~/.local/bin/virtualenvwrapper.sh
VENV=fairseq-mods20210419
echo "Activating $VENV ..."
workon $VENV

EXPERIMENT_DIR=$HOME/tagged-seq2seq
echo "cd $EXPERIMENT_DIR"
cd $EXPERIMENT_DIR

TEXT="$INPUT_DIR/${TOK:+$TOK.}$TAG${ABLATION:+.$ABLATION}"
OUTDIR="data-bin/$OUTPUT_PREFIX-${TOK:+$TOK-}$TAG${ABLATION:+-$ABLATION}"

# using a shared dictionary
if [[ $SHARED == "shared" ]]; then
  OUTDIR=$OUTDIR-shared$ADDITIONAL_OPTIONS
  if [[ $ABLATION != "" ]]; then
    echo "Shared token dictionary currently not implemented for asymmetric tagging"
    echo "Try without \"shared\" or without \"srconly/tgtonly\""
    exit 1
  else
    ADDITIONAL_OPTIONS="--joined-dictionary $ADDITIONAL_OPTIONS"
  fi
else
  OUTDIR=$OUTDIR$ADDITIONAL_OPTIONS
fi

# binarizing all splits (with dict from train split)
# or reusing existing dict for NE subsplits
if [[ ($SUBSPLIT == "someNE") || ($SUBSPLIT == "noNE") || ($ABLATION != "")]]; then
  if [[ $ABLATION == "tgtonly" ]]; then
    SRCDICT=notags
  else
    SRCDICT=$TAG
  fi
  if [[ $ABLATION == "srconly" ]]; then
    TGTDICT=notags
  else
    TGTDICT=$TAG
  fi

  DICTS="--srcdict data-bin/$OUTPUT_PREFIX-${TOK:+$TOK-}$SRCDICT${SHARED:+-shared}/dict.$SRC.txt"
  if [[ ($SHARED != "shared") || ($ABLATION != "") ]]; then
    DICTS="$DICTS --tgtdict data-bin/$OUTPUT_PREFIX-${TOK:+$TOK-}$TGTDICT${SHARED:+-shared}/dict.$TGT.txt"
  fi
  OUTDIR=$OUTDIR${SUBSPLIT:+-$SUBSPLIT}
else
  DICTS=""
fi

if [[ ($SUBSPLIT == "someNE") || ($SUBSPLIT == "noNE") ]]; then
  SPLITS="--testpref $TEXT.test.$SUBSPLIT"
else
  SPLITS="--trainpref $TEXT.train --validpref $TEXT.valid --testpref $TEXT.test"
fi

# echo PREPROCESSING PARAMS
# echo "TEXT=$TEXT"
# echo "SRC=$SRC"
# echo "TGT=$TGT"
# echo "OUTDIR=$OUTDIR"
# echo "DICTS=$DICTS"
# echo "SPLITS=$SPLITS"
# echo "TAG_SEP=$TAG_SEP"
# echo "ADDITIONAL_OPTIONS=$ADDITIONAL_OPTIONS"

# export

binarize="TAG_SEP=\"$TAG_SEP\" fairseq-preprocess \
--user-dir tagged-seq2seq --task tagged_translation \
--source-lang $SRC --target-lang $TGT \
$DICTS \
$SPLITS \
--workers 20 \
--destdir $OUTDIR \
$ADDITIONAL_OPTIONS"

echo
echo Preprocessing command:
echo $binarize
echo

eval $binarize

deactivate
