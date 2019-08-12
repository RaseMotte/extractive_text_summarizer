# Step 1, load the data

import os
import struct
import tensorflow as tf

def as_tensors(bin_path):
  """
  \brief  Parse the binary file into a record.

  \example
    parsed_chunk = as_tensors('path/to/data.bin')
    for parsed_record in parsed_chunk.take(10):
      target_sum = parsed_record["abstract"]
      art = parsed_record["article"]
  """
  raw_chunk = tf.data.TFRecordDataset(bin_path)

  # Create a description of the features.
  feature_description = {
      'article': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'abstract': tf.io.FixedLenFeature([], tf.string, default_value=''),
  }

  def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

  # apply for each entry the parsing function
  parsed_chunk = raw_chunk.map(_parse_function)
  return parsed_chunk


def stringGenerator(bin_path):
  """
  """
  with open(bin_path, "rb") as reader:
      eof = False
      while not eof:
          len_bytes = reader.read(8)
          if not len_bytes:
              eof = True
              break
          str_len = struct.unpack('q', len_bytes)[0]
          example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
          #print(example_str)
          tf_example = tf.train.Example.FromString(example_str)
          art_byte_rpz = str(tf_example.features.feature['article'].bytes_list.value)
          abs_byte_rpz = str(tf_example.features.feature['abstract'].bytes_list.value)
          yield art_byte_rpz, abs_byte_rpz
