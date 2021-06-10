#!/bin/bash

echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

echo
echo Job starting ...
echo "node: $(/bin/hostname)"
date

DATASET=$1
CHECKPOINT_SUFFIX=$2

SUBSPLIT=${3:+-$3}

ADDITIONAL_OPTIONS=$4

EPOCH=${5:-40}

BATCH_MAX=512

DATADIR=data-bin/$DATASET$SUBSPLIT
CHECKPOINT=checkpoints/$DATASET$CHECKPOINT_SUFFIX/checkpoint$EPOCH.pt
RESULTSDIR=results/$DATASET$CHECKPOINT_SUFFIX$SUBSPLIT-chkpt$EPOCH$ADDITIONAL_OPTIONS

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

evaluate="fairseq-generate $DATADIR \
--user-dir tagged-seq2seq --task tagged_translation \
--scoring bleu --remove-bpe sentencepiece \
--path $CHECKPOINT \
--results-path $RESULTSDIR \
--max-tokens $BATCH_MAX \
$ADDITIONAL_OPTIONS"

echo
echo Evaluation command:
echo $evaluate
echo

eval $evaluate
STATUS=$?

sleep 1 # flush buffers

if [[ $STATUS != 0 ]]; then
  echo
  echo "Error during evaluation ..."
  date
  exit $STATUS
fi

echo
echo "Result: $RESULTSDIR/generate-test.txt"
echo "Checkpoint ${CHECKPOINT#checkpoints/} on dataset ${DATADIR#data-bin/}"
tail -n1 $RESULTSDIR/generate-test.txt

echo
echo "##################"
echo Evaluation finished!
date

deactivate
