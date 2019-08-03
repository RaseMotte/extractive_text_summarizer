This is a end of studies' project, realized with Exensa.

# 1. Packages

Each of our packages have its own README.md and requirements.txt so the project is modulable.

If you wish to install dependencies for the whole project:
```sh
pip install --user -r requirements.txt
```

# Dataset preprocessing | ```./preprocessing```

Source code and more information [available here](https://github.com/becxer/cnn-dailymail/).

## Content
* Instructions on how to download dataset.
* Instructions on how to use the preprocessing scipt.

## Overview
* Language detection **TODO**
* Tokenization with Stanford CoreNLP PTBTokenizer **TODO** Tokenizer as variable
* Split in train, test and validation sets.
* Binarization with TensorFlow