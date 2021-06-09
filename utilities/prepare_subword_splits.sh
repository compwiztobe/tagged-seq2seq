# after building splits with prepare_tagged_splits.sh
# now ready to train sentence piece and apply to all

utilities/tokenize.sh $PREFIX.3m $TAG $SRC $TGT $TOK $SIZE
