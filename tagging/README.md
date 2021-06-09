## Flair tagging

Requires [Flair](https://github.com/flairNLP/flair) v0.8.0.post1 (with PyTorch plus CUDA/cuDNN for tagging on a GPU):
```$ pip install -r requirements.txt```

To tag a corpus file (one sentence per line):
```$ python tag.py -m [ner,upos]-multi [--ner-stats] --print < input_file > tagged_file```
