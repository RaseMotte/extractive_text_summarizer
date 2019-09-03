import os
import sys
from make_datafiles import num_expected_cnn_stories, num_expected_dm_stories
from make_datafiles import tokenize_stories, save_datafiles_in_chunks
from utils import check_num_stories

SRC_DIR, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR, _ = os.path.split(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data/cnn-dailymail")

CNN_TOKENIZED_STORIES_DIR = os.path.join(DATA_DIR, "cnn_stories_tokenized_v2")
DM_TOKENIZED_STORIES_DIR = os.path.join(DATA_DIR, "dm_stories_tokenized_v2")

FINISHED_FILES_DIR = os.path.join(DATA_DIR, "finished_files_v4")
CHUNKS_DIR = os.path.join(FINISHED_FILES_DIR, "chunked")

ALL_TRAIN_URLS = os.path.join(DATA_DIR, "url_lists/all_train.txt")
ALL_VAL_URLS = os.path.join(DATA_DIR, "url_lists/all_val.txt")
ALL_TEST_URLS = os.path.join(DATA_DIR, "url_lists/all_test.txt")

CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print("USAGE: python make_datafiles.py <cnn_stories_dir> <dailymail_stories_dir>")
    sys.exit()
  cnn_stories_dir = sys.argv[1]
  dm_stories_dir = sys.argv[2]

  # Check the stories directories contain the correct number of .story files
  check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
  check_num_stories(dm_stories_dir, num_expected_dm_stories)

  # Create some new directories
  if not os.path.exists(CNN_TOKENIZED_STORIES_DIR):
    os.makedirs(CNN_TOKENIZED_STORIES_DIR)
  if not os.path.exists(DM_TOKENIZED_STORIES_DIR):
    os.makedirs(DM_TOKENIZED_STORIES_DIR)
  if not os.path.exists(CHUNKS_DIR):
    os.makedirs(CHUNKS_DIR)

  # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  tokenize_stories(cnn_stories_dir, CNN_TOKENIZED_STORIES_DIR)
  tokenize_stories(dm_stories_dir, DM_TOKENIZED_STORIES_DIR)

  # Generates the oracle summary.
  # Saves the result in chunks of 100 containing preprocessed article/abstrac and oracle ids.
  token_dirs = [CNN_TOKENIZED_STORIES_DIR, DM_TOKENIZED_STORIES_DIR]
  save_datafiles_in_chunks(token_dirs, CHUNKS_DIR, CHUNK_SIZE, ALL_TRAIN_URLS)
  save_datafiles_in_chunks(token_dirs, CHUNKS_DIR, CHUNK_SIZE, ALL_TEST_URLS)
  save_datafiles_in_chunks(token_dirs, CHUNKS_DIR, CHUNK_SIZE, ALL_VAL_URLS)