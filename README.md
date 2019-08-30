This is a end of studies' project, realized with Exensa.

# 1. Packages

Install dependencies for the whole project:
```sh
pip install --user -r requirements.txt
```

# 2. Architecture

## Data | ```./data```
* Datasets
* Tokenized files
* Binarized files

## Intermediate results | ```./model```
Folder for storing binary (json or other format) file for local use.

* model metadata: date, size of training data, ect **FIXME**

## Dataset preparation and processing | ```./src/preprocessing```

Source code and more information [available here](https://github.com/becxer/cnn-dailymail/).

### Content
* Instructions on how to download dataset.
* Instructions on how to use the preprocessing scipt.

### Overview
* Language detection **TODO**
* Sentence and word tokenization with Stanford CoreNLP
* Split in train, test and validation sets.
* Binarization with TensorFlow

Produced files are placed in ./data/cnn-dailymail/finished_files_v2/.


## Model structure, training and evaluation | ```./src/modeling```

* All model tested or to test.
* A logger
* Tools for benchmarking

### Models

* Gensim **FIXME**


---
# TODO

- [x] Data writer from tokenized files to binary
- [ ] Data reader from .bin for model
- [ ] Logger
- [ ] Benchmarks
- [ ] Gensim summarizer