#!/bin/bash

# slurm compat
# cd $SLURM_SUBMIT_DIR

# echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

echo
echo Job starting ...
#srun -l echo "node: $(/bin/hostname)"
#srun -l echo "cwd: $(/bin/pwd)"
#srun -l /bin/date
echo "node: $(/bin/hostname)"
# echo "cwd: $(/bin/pwd)"
date

DATASET=$1
#BATCH_MAX=$2
#BATCHES_PER_UPDATE=$3
#WARMUP_INTERVAL=$4

BATCH_MAX=512
BATCHES_PER_UPDATE=16
WARMUP_INTERVAL=500

ADDITIONAL_OPTIONS=$2

EPOCHS=${3:-40}

GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')
UPDATE_FREQ=$(expr $BATCHES_PER_UPDATE / $GPU_COUNT)

DATADIR=data-bin/$DATASET
LOGDIR=logs/$DATASET$ADDITIONAL_OPTIONS
CHECKPOINTDIR=checkpoints/$DATASET$ADDITIONAL_OPTIONS

echo
echo EXPERIMENT PARAMS
echo "DATADIR=$DATADIR"
echo "LOGDIR=$LOGDIR"
echo "CHECKPOINTDIR=$CHECKPOINTDIR"
echo "BATCH_MAX=$BATCH_MAX"
echo "BATCHES_PER_UPDATE=$BATCHES_PER_UPDATE"
echo "(GPU_COUNT=$GPU_COUNT)"
echo "(UPDATE_FREQ=$UPDATE_FREQ)"
echo "WARMUP_INTERVAL=$WARMUP_INTERVAL"
echo "EPOCHS=$EPOCHS"

echo "ADDITIONAL_OPTIONS=$ADDITIONAL_OPTIONS"

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

train="fairseq-train $DATADIR \
--user-dir tagged-seq2seq --task tagged_translation --arch tagged_transformer \
--save-dir $CHECKPOINTDIR \
--tensorboard-logdir $LOGDIR \
--share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr 5e-4 --lr-scheduler inverse_sqrt --dropout 0.3 --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--eval-bleu --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
--eval-bleu-detok space --eval-bleu-remove-bpe sentencepiece --eval-bleu-print-samples \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--max-tokens $BATCH_MAX \
--update-freq $UPDATE_FREQ \
--warmup-updates $WARMUP_INTERVAL \
--fp16 \
--max-epoch $EPOCHS \
--no-epoch-checkpoints \
--keep-best-checkpoints 1 \
$ADDITIONAL_OPTIONS"

echo
echo Training command:
echo $train
echo

eval $train
STATUS=$?

sleep 1 # just in case of stderr flushing or something

if [[ $STATUS != 0 ]]; then
  echo
  echo "Error during training ..."
  date
  exit $STATUS
fi

echo
if [ -a $CHECKPOINTDIR/checkpoint$EPOCHS.pt ]; then
  echo "$CHECKPOINTDIR/checkpoint$EPOCHS.pt already exists, not overwriting with new checkpoint_last.pt"
else
  echo -n "cp "
  cp -nv $CHECKPOINTDIR/checkpoint_last.pt $CHECKPOINTDIR/checkpoint$EPOCHS.pt
fi

echo
echo "##################"
echo Training finished!
date

deactivate

# slurm  job status
# squeue  --job  $SLURM_JOBID

# echo  "##### END #####"
