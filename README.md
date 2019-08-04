This is a end of studies' project, realized with Exensa.

# 1. Packages

Each of our packages have its own README.md and requirements.txt so the project is modulable.

If you wish to install dependencies for the whole project:
```sh
pip install --user -r requirements.txt
```

# 2. Architecture

## data | ```./data```
* Datasets
* Tokenized files
* Binarized files

## Intermediate results | ```./model```
Folder for storing binary (json or other format) file for local use.

* model metadata: date, size of training data, ect **FIXME**

## Dataset preparation and processing | ```./src/preprocessing```

Source code and more information [available here](https://github.com/becxer/cnn-dailymail/).

## Content
* Instructions on how to download dataset.
* Instructions on how to use the preprocessing scipt.

## Overview
* Language detection **TODO**
* Tokenization with Stanford CoreNLP PTBTokenizer **TODO** Tokenizer as variable
* Split in train, test and validation sets.
* Binarization with TensorFlow

Produced files are placed in model.


## Model structure, training and evaluation | ```./src/modeling```

* Model structure interface that must be implemented by any new model structure.
* All model tested or to test.

### Models