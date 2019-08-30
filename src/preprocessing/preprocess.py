import os
import sys
from data_builder import num_expected_cnn_stories, num_expected_dm_stories
from data_builder import check_num_stories, tokenize_stories, write_to_bin, chunk_all, gen_oracle_summary
from data_builder import CNN_TOKENIZED_STORIES_DIR, DM_TOKENIZED_STORIES_DIR, FINISHED_FILES_DIR
from data_builder import ALL_TEST_URLS, ALL_TRAIN_URLS, ALL_VAL_URLS

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
  if not os.path.exists(FINISHED_FILES_DIR):
    os.makedirs(FINISHED_FILES_DIR)

  # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  tokenize_stories(cnn_stories_dir, CNN_TOKENIZED_STORIES_DIR)
  tokenize_stories(dm_stories_dir, DM_TOKENIZED_STORIES_DIR)

  # Read the tokenized stories, do a little postprocessing, generate the summary, then write to bin files
  write_to_bin(ALL_TEST_URLS, os.path.join(FINISHED_FILES_DIR, "test.bin"))
  write_to_bin(ALL_VAL_URLS, os.path.join(FINISHED_FILES_DIR, "val.bin"))
  write_to_bin(ALL_TRAIN_URLS, os.path.join(FINISHED_FILES_DIR, "train.bin"), makevocab=True)

  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunk_all()
