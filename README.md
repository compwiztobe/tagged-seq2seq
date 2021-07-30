# Tag Assisted Neural Machine Translation

<img align="right" src="./diagram.svg" title="Tagged seq2seq" width="360">

This is the source code for results published in [Siekmeier et al., 2021](https://aclanthology.org/2021.iwslt-1.30).

This repo consists of the following:
- `tagged-seq2seq/` - the main fairseq extensions to be imported with `--user-dir`
   for data binarization, model training, and evaluation;
as well as:
- `tagging/` - [Flair](https://github.com/flairNLP/flair)
   tagging scripts to add tags to raw data
- `utilities/` - text preprocessing utilities to prepare datasets

## Getting started

### Initialize dataset

IWSLT'14:
```
$ utilities/prepare_iwslt14.sh
```

OpenSubtitles:
```
$ pip install opustools
$ utilities/prepare_opensub.sh de en
```

These prepare a single data file for tagging.

### Tagging with Flair

See `./tagging/README.md` for more details.

### Preprocessing

Collecting tagged sentences into splits, tokenizing, covering all factors
```
$ pip install -r utilities/requirements.txt
$ utilities/prepare_tagged_splits.sh opus/opensub.de-en 3m ner de en "$splits"
$ utilities/tokenize.sh opus/opensub.de-en.3m de en unigram 32k
```
To generate new OpenSubs splits, use, for example, `splits="train,3000000 valid,100000 test,100000"`.

For existing IWSLT14 splits, `splits="train,160239 valid,7283 test,6750 --consecutive"` (remove `--consecutive` to rerandomize).

See `./utilities/README.md` for more details.

### Model training and evaluation

Requires CUDA, cuDNN, PyTorch, and [fairseq](https://github.com/pytorch/fairseq) v0.10.2
```
pip install -r requirements.txt
SITEPACKAGES=$(python -c "import fairseq as _; print(_.__path__[0].rsplit('/',1)[0])") &&
patch -d $SITEPACKAGES -p1 < 0001-patch-for-fairseq-v0.10.2-to-apply-in-fairseq-venv-s.patch
```
to install dependencies and required fairseq patches.

Binarization requires these dependencies and code, but no GPU.
```
./binarize.sh opus/opensub.de-en.3m de-en unigram32k ner OpenSub-3m shared
```

_(Requires CUDA-enabled GPU)_ To train and evaluate a new model, specify dataset
(e.g. `IWSLT14-unigram32k-notags`, `OpenSub-3m-unigram32k-ner`):
```
export CUDA_VISIBLE_DEVICES=...
./train.sh transformer OpenSub-3m-unigram32k-ner-shared "--share-all-embeddings"
./eval.sh OpenSub-3m-unigram32k-ner-shared "-transformer--share-all-embeddings"
```

In place of `ner`, can also run `ner-srconly`, `ner-tgtonly`, `upos`, `notags`, etc.

Tensorboard logs are written to `logs/dataset`, fairseq-generate output in `results/dataset`.

## Citation

(Siekmeier et al., 2021)

Aren Siekmeier, WonKee Lee, Hongseok Kwon, and Jong-Hyeok Lee. 2021. [Tag assisted neural machine translation of film subtitles](https://doi.org/10.18653/v1/2021.iwslt-1.30). In _Proceedings of the 18th International Conference on Spoken Language Translation (IWSLT 2021)_, pages 255–262, Bangkok, Thailand (online). Association for Computational Linguistics.

``` bibtex
@inproceedings{siekmeier-etal-2021-tag,
    title = "Tag Assisted Neural Machine Translation of Film Subtitles",
    author = "Siekmeier, Aren  and
      Lee, WonKee  and
      Kwon, Hongseok  and
      Lee, Jong-Hyeok",
    booktitle = "Proceedings of the 18th International Conference on Spoken Language Translation (IWSLT 2021)",
    month = aug,
    year = "2021",
    address = "Bangkok, Thailand (online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.iwslt-1.30",
    doi = "10.18653/v1/2021.iwslt-1.30",
    pages = "255--262",
    abstract = "We implemented a neural machine translation system that uses automatic sequence tagging to improve the quality of translation. Instead of operating on unannotated sentence pairs, our system uses pre-trained tagging systems to add linguistic features to source and target sentences. Our proposed neural architecture learns a combined embedding of tokens and tags in the encoder, and simultaneous token and tag prediction in the decoder. Compared to a baseline with unannotated training, this architecture increased the BLEU score of German to English film subtitle translation outputs by 1.61 points using named entity tags; however, the BLEU score decreased by 0.38 points using part-of-speech tags. This demonstrates that certain token-level tag outputs from off-the-shelf tagging systems can improve the output of neural translation systems using our combined embedding and simultaneous decoding extensions.",
}
```
