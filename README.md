Running these extensions to fairseq:

```
pip install torch==1.7.1
git clone -b v0.10.2 https://github.com/pytorch/fairseq@
pip install --editable fairseq
git clone https://github.com/compwiztobe/tagged-seq2seq
cd tagged-seq2seq
fairseq-... --user-dir . --task tagged_translation
'''
