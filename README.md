#### Tagged seq2seq data preparation tools

To maintain separation of concerns, data preparation wherever possible is done
without interacting with the seq2seq model codebase, fairseq or Flair dependencies,
or systems requiring CUDA or GPU processing.  The preprocessing pipeline expects
an initial parallel corpus in a single line containing sentence pairs on every
two lines.  Since tagging with Flair requires a GPU and is also a comparatively
heavy operation, we attempt to put this as early as possible in the pipeline
so that tagged output can be reused by all later data split preprocessing steps.
SentencePiece as a dependency and a training and tokenization operation
are fairly lightweight so these come later.  Finally, building the binary dataset
for fairseq input is left to the last step and requires the same codebase used
for running the model itself.

Where indicated (steps 1, 3, 5), it is recommended to separate virtual environments
for the dependencies needed for those operations (opustools or Flair or fairseq).
These dependencies are specified in the requirements files and READMEs in
`../tagging` and `../tagged-seq2seq`.

For the other preprocessing steps here, use this directory's `requirements.txt`:

```
pip install sentencepiece
```

##### Data prep pipeline overview:

1. Retrieving data

Collect parallel corpus from source, one-to-one sentence alignments with each
pair on every two lines of a single file.  For OpenSubtitles data from OPUS,
`pip install opustools` and `opus_read ...`.  Other data sources will vary.

2. Initial preprocessing

    - Length filtering with
      `filter.py MIN_LEN MAX_LEN MAX_RATIO < pairfile > filtered_pairfile`
    - Lang splitting with `split_langs.sh pairfile SRC TGT`
      (or merge this with `split_even_odd.sh` logic used later)

3. Tagging with Flair

This is done with a separate code base up in `../tagging`.  Requires a CUDA device.

4. Final preprocessing

    - Split generation:
      `random_splits.py prefix linecount 2 train,train_size valid,valid_size test,test_size`
    - Find test subsplits `{some,no}NE` with `ne_split.py prefix sep 2`
    - Copying tagged lines according to `split.lang.idx` to create `srconly`,
      `tgtonly`, and `tagged` splits. (This script doesn't exist).
    - Splitting all split files with `split_even_odd.sh pairfile SRC TGT`
      (this one could instead have input piped to it, so consider that).
    - Now we need to tokenize.  Train tokenizer with
      `train_tokenizer.py --vocab-size VOCAB_SIZE --model-type {unigram,bpe,char,word} --model-prefix OUTPUT`.
    - Tokenize all split files using that trained sentencepiece model,
      with tag broadcasting.

5. Fairseq binarization

With the `../src` code and pip requirements, `fairseq-preprocess` on train valid test
for all corners of the tagtype, srctag, tgttag cube, plus on `test.{some,noNE}`
once dictionaries are built.  This step does NOT require a CUDA device
(though later training will).
