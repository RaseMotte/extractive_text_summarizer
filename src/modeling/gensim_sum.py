import os
import sys
import struct
from gensim.summarization.summarizer import summarize
from tensorflow.core.example import example_pb2
from rouge import Rouge

from benchmark import log_time
from load_datafiles import stringGenerator
from logger import ModelLogger

GENSIM_DATA_DIR = '../../model/gensim_sum'

gensim_logger = ModelLogger("gensim_sum")

def get_ratio(text):
    """
    Return the % of sentence to keep for the summary to be 4 sentences long.
    """
    n = text.count('.')
    r = 3
    pn = 1
    pr = (pn * r) / n
    return pr


@log_time(gensim_logger)
def transform(data_path):
    str_generator = stringGenerator(data_path)
    rouge = Rouge()

    _, record_name = os.path.split(data_path)
    summaries_file = os.path.join(GENSIM_DATA_DIR, 'summaries_%s' % record_name)

    with open(summaries_file, 'wb') as writer:
        record_i = 0
        for article, target_sum in str_generator:
            pr = get_ratio(article)
            if article.count('.') <= 13:
                print("=============================================================================================================================")
                print(article.count('.'))
                print('\n')
                print(article)
                tmp = article.split('.')
                for i, p in enumerate(tmp):
                    print("%d : %s\n" % (i, p))
            assert(pr != 0)
            out_sum = summarize(article, pr)
            if article.count('.') <= 13:
                print('RES: \n')
                print(out_sum)
                print('\nTARGET: \n')
                print(target_sum)
            json_rouge = rouge.get_scores(out_sum, target_sum)
            gensim_logger.rouge_debug(data_path, record_i, json_rouge)
            record_i += 1

            tf_example = example_pb2.Example()
            tf_example.features.feature['produced_sum'].bytes_list.value.extend([out_sum.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: python gensim_sum.py <binary_file_dir>")
        sys.exit()
    data_path = sys.argv[1]

    if os.path.isdir(data_path):
        files = os.listdir(data_path)
        for f in files:
            transform(f)
    else:
        transform(data_path)
    print("Done")