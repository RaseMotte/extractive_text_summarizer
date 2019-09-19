import sys
import os

from load_datafiles import load_chunk_corpus
from eval_tools import *
from GensimSum import GensimSum, GENSIM_DATA_DIR

# List of modules containing models to evaluate with they data dir
MODELS = [(GensimSum, GENSIM_DATA_DIR)]

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("USAGE: python eval_models.py <path/to/chunk/json/story>")
    sys.exit(1)
  chunk_file = sys.argv[1]
  if not os.path.exists(chunk_file):
    print("ERROR: %s does not exist." % chunk_file)
    sys.exit(1)

  h, art, target_sum = load_chunk_corpus(chunk_file)

  for ModelSum, out_dir in MODELS:
    model_sum = ModelSum()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    batch_eval_model(model_sum, h, art, target_sum, out_dir)
