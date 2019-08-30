import collections
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import torch

import tensorflow as tf
from tensorflow.core.example import example_pb2

from utils import _get_ngrams, _get_word_ngrams, cal_rouge

SRC_DIR, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR, _ = os.path.split(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data/cnn-dailymail")

ALL_TRAIN_URLS = os.path.join(DATA_DIR, "url_lists/all_train.txt")
ALL_VAL_URLS = os.path.join(DATA_DIR, "url_lists/all_val.txt")
ALL_TEST_URLS = os.path.join(DATA_DIR, "url_lists/all_test.txt")

CNN_TOKENIZED_STORIES_DIR = os.path.join(DATA_DIR, "cnn_stories_tokenized_v2")
DM_TOKENIZED_STORIES_DIR = os.path.join(DATA_DIR, "dm_stories_tokenized_v2")
FINISHED_FILES_DIR = os.path.join(DATA_DIR, "finished_files_v2")

CHUNKS_DIR = os.path.join(FINISHED_FILES_DIR, "chunked")


dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote,
              dm_double_close_quote, ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data


def chunk_file(set_name):
  in_file = os.path.join(FINISHED_FILES_DIR, ('%s.bin' % set_name))
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(CHUNKS_DIR, '%s_%03d.bin' %
                               (set_name, chunk))  # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(CHUNKS_DIR):
    os.mkdir(CHUNKS_DIR)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % CHUNKS_DIR)


def tokenize_stories(stories_dir, tokenized_stories_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print("Preparing to tokenize %s to %s..." %
        (stories_dir, tokenized_stories_dir))
  stories = os.listdir(stories_dir)
  # make IO list file
  print("Making list of files to tokenize...")
  with open("mapping.txt", "w") as f:
    for s in stories:
      f.write("%s\n" % (os.path.join(stories_dir, s)))
  command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
             '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping.txt', '-outputFormat', 'json', '-outputDirectory', tokenized_stories_dir]
  print("Tokenizing %i files in %s and saving in %s..." %
        (len(stories), stories_dir, tokenized_stories_dir))
  subprocess.call(command)
  print("Stanford CoreNLP Tokenizer has finished.")
  os.remove("mapping.txt")

  # Check that the tokenized stories directory contains the same number of files as the original directory
  num_orig = len(os.listdir(stories_dir))
  num_tokenized = len(os.listdir(tokenized_stories_dir))
  if num_orig != num_tokenized:
    raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
        tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
  print("Successfully finished tokenizing %s to %s.\n" %
        (stories_dir, tokenized_stories_dir))


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


REMAP = {"-lbr-": "(", "-rbr-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s.encode())
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line:
    return line
  if line == "":
    return line
  if line[-1] in END_TOKENS:
    return line
  # print line[-1]
  return line + " ."


def clean(x):
    return fix_missing_period(re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x))


def get_art_abs_from_text(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx, line in enumerate(lines):
    if line == "":
      continue  # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  return article_lines, highlights


def get_art_abs_from_json(json_file):
  article_tokens_list = []
  abstract_tokens_list = []
  flag = False
  for sent in json.load(open(json_file))['sentences']:
      tokens = [t['word'] for t in sent['tokens']]
      tokens = [t.lower() for t in tokens]
      if (tokens[0] == '@highlight'):
          flag = True
          continue
      if (flag):
          abstract_tokens_list.append(tokens)
          flag = False
      else:
          article_tokens_list.append(tokens)
  article_tokens_list = [clean(' '.join(sent)).split()
                         for sent in article_tokens_list]
  abstract_tokens_list = [clean(' '.join(sent)).split()
                          for sent in abstract_tokens_list]

  return article_tokens_list, abstract_tokens_list


def greedy_selection(article_tokens_list, abstract_tokens_list, summary_size):
    """
    From an abstract summary, generate an extractive summary for the given article.

    :param article_tokens_list:   List of sentences representing a document.
                            Each sentence is a list of word.
                            Sentence are here tokenized by Stanford Core NLP.

    :param abstract_tokens_list:  List of sentence representing the abstract given for the document.
                                  Format is the same as article_tokens_list.

    :returns:         An array of index pointing to the selected sentences in article_tokens_list.
    """
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_tokens_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in article_tokens_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def gen_oracle_summary(story_file):
  """
    Re-write the abstract summaries to extractive summaries using a greedy algorithm.
  """
  article_tokens_list, abstract_tokens_list = get_art_abs_from_json(story_file)
  oracles_ids = greedy_selection(article_tokens_list, abstract_tokens_list, 3)
  return article_tokens_list, abstract_tokens_list, oracles_ids


def write_to_bin(url_file, out_file, makevocab=False):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print("Making bin file for URLs listed in %s..." % url_file)
  url_list = read_text_file(url_file)
  url_hashes = get_url_hashes(url_list)
  story_fnames = [s+".story.json" for s in url_hashes]
  num_stories = len(story_fnames)

  if makevocab:
    vocab_counter = collections.Counter()

  with open(out_file, 'wb') as writer:
    for idx, s in enumerate(story_fnames):
      if idx % 1000 == 0:
        print("Writing story %i of %i; %.2f percent done" %
              (idx, num_stories, float(idx)*100.0/float(num_stories)))

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(CNN_TOKENIZED_STORIES_DIR, s)):
        story_file = os.path.join(CNN_TOKENIZED_STORIES_DIR, s)
      elif os.path.isfile(os.path.join(DM_TOKENIZED_STORIES_DIR, s)):
        story_file = os.path.join(DM_TOKENIZED_STORIES_DIR, s)
      else:
        print("Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (
            s, CNN_TOKENIZED_STORIES_DIR, DM_TOKENIZED_STORIES_DIR))
        # Check again if tokenized stories directories contain correct number of files
        print("Checking that the tokenized stories directories %s and %s contain correct number of files..." % (
            CNN_TOKENIZED_STORIES_DIR, DM_TOKENIZED_STORIES_DIR))
        check_num_stories(CNN_TOKENIZED_STORIES_DIR, num_expected_cnn_stories)
        check_num_stories(DM_TOKENIZED_STORIES_DIR, num_expected_dm_stories)
        raise Exception("Tokenized stories directories %s and %s contain correct number of files but story file %s found in neither." % (
            CNN_TOKENIZED_STORIES_DIR, DM_TOKENIZED_STORIES_DIR, s))

      # Get the strings to write to binary: abstact is the abstraction-based summary and
      # oracles ids refer to sentences forming the target extractive summary.
      article_tokens_list, abstract_tokens_list, oracles_ids = gen_oracle_summary(
          story_file)

      # Make abstract and article into a signle string, putting <s> and </s> tags around the sentences
      article_sent_list = [' '.join(token_list)
                           for token_list in article_tokens_list]
      article = ' '.join(
          ["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in article_sent_list])

      abstract_sent_list = [' '.join(token_list)
                            for token_list in abstract_tokens_list]
      abstract = ' '.join(
          ["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in abstract_sent_list])

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([
                                                                     article.encode()])
      tf_example.features.feature['abstract'].bytes_list.value.extend([
                                                                      abstract.encode()])
      tf_example.features.feature['oracle_ids'].int64_list.value.extend(
          oracles_ids)
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(' ')
        abs_tokens = abstract.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [
            SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens]  # strip
        tokens = [t for t in tokens if t != ""]  # remove empty
        vocab_counter.update(tokens)

  print("Finished writing file %s\n" % out_file)

  # write vocab to file
  if makevocab:
    print("Writing vocab file...")
    with open(os.path.join(FINISHED_FILES_DIR, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")


def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (
        stories_dir, num_stories, num_expected))
