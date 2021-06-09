## Tagged seq2seq data preparation tools

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

Where indicated (steps 1, 3, 6), it is recommended to separate virtual environments
for the dependencies needed for those operations (opustools or Flair or fairseq).
These dependencies are specified in the requirements files and READMEs in
`../tagging` and `../tagged-seq2seq`.  Step 5 requires SentencePieces and
2 and 4 require only Python 3.8.

For the other preprocessing steps here, use this directory's `requirements.txt`:

```
pip install sentencepiece==0.1.95
```

### Data prep pipeline overview:

1. Retrieving data

Collect parallel corpus from source, one-to-one sentence alignments with each
pair on every two lines of a single file.  For OpenSubtitles data from OPUS,
`pip install opustools` and `opus_read ...`.  Other data sources will vary.

2. Initial preprocessing

    - Length filtering with
      ```$ filter.py min_len max_len max_ratio < pairfile > filtered_pairfile```
    - Lang splitting with `split_langs.sh pairfile src tgt`

Alternatively, data preparation scripts such as `fairseq/examples/translation/prepare_iwslt14.sh`
will prepare some temporary files in a form ready for step 3
(word tokenized, not subword tokenized, for tagging, and
one sentence on each line of a separate file for each lang and split,
words and phrase-ending punctuation - , . etc. - separated by space)

To prepare IWSLT14 data into a single source file for each language:
```$ cat [iwslt14path]/tmp/{train,valid,test}.src > iwslt14/iwslt14.src-tgt.src```
```$ cat [iwslt14path]/tmp/{train,valid,test}.tgt > iwslt14/iwslt14.src-tgt.tgt```
and pass `--consecutive` as an additional arg along with the split sizes (e.g. from `wc`)
to `generate_splits.sh` below to prevent split randomization.  Then generate splits
can also generate the index files needed for copying data to other splits with `copy_splits.sh`.

3. Tagging with Flair

This is done with a separate code base up in `../tagging`.  Requires a CUDA device.  See `../tagging/README.md`

4. Split generation

    Requires only Python 3.8.

    - Tagged split generation:
      ```$ generate_splits.sh prefix suffix tag srclang tgtlang "train,train_size valid,valid_size test,test_size" additional_args```
    - Copying tagged lines according to `split.lang.idx` to create `srconly`,
      `tgtonly`, and `tagged` splits.
      ```$ copy_splits.sh prefix suffix tag srclang tgtlang "train,train_size, valid,valid_size test,test_size"```
    - (For NER tagging) Find test subsplits `{some,no}NE` with `ne_split.py prefix sep 2`

5. Subword tokenization

    Requires `pip install sentencepiece==0.1.95`)

    Now we need to tokenize.  Train tokenizer and apply to all splits with
    ```$ tokenize.sh datadir srclang tgtlang tok_type vocab_size```
    where `TOK_TYPE` can be `unigram`, `bpe`, `word`, `char`

6. Fairseq binarization

With the `../tagged-seq2seq` code files and pip requirements, `fairseq-preprocess` on train valid test
for all corners of the tagtype, srctag, tgttag cube, plus on `test.{some,noNE}`
once dictionaries are built.  This step does NOT require a CUDA device
(though later training will).
