PREFIX=$1
SRC_LANG=$2
TGT_LANG=$3
ARGS=${@:4:}

# preprocess token files
fairseq-preprocess --source-lang "$SRC_LANG" --target-lang "$TGT_LANG" \
  --trainpref "$PREFIX"
  --destdir
