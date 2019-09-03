import os
import torch

from gensim.summarization.summarizer import summarize
from rouge import Rouge

from BaseModel import BaseModel

GENSIM_DATA_DIR = '../../model/gensim_sum_tmp_test'

gensim_logger = ModelLogger("gensim_sum")

def _get_ratio(text):
  """
  Return the % of sentence to keep for the summary to be r sentences long.
  """
  n = text.count('.')
  if n == 0:
    return 1
  r = 3
  pn = 1
  pr = (pn * r) / n
  return pr

class GensimSum(BaseModel):
  """
  Gensim summariser based on TextRank.
  """

  def __init__(self, name='GensimSum'):
    super(GensimSum, self).__init__(name)

  def transform(self, train_x):
    pred_sum = []
    for aid, article in enumerate(train_x):
      pr = _get_ratio(article)
      try:
        assert(pr != 0)
        out_sum = summarize(article, pr)
      except (ValueError, AssertionError):
        self.fail.append(aid)
        print("\n===================================================================================================================================================")
        print ("REMOVED")
        print(article)
        continue
      pred_sum.append(out_sum)
    return pred_sum

  def score(self, test_x, test_y):
    """
    Computes the mean rouge scores between predicted summaries and
    target summaries.
    """
    pred_y = self.transform(test_x)
    for aid in self.fail:
      test_x = test_x.pop(aid)
      test_y = test_y.pop(aid)
    rouge = Rouge()
    self.mean_r_scores = rouge.get_scores(pred_y, test_y, avg=True)
    return self.mean_r_scores

  def report(self, pred_y, test_y):
    if self.mean_r_scores is None:
      rouge = Rouge()
      self.mean_r_scores = rouge.get_scores(pred_y, test_y, avg=True)
    #r_scores = rouge.get_scores(pred_y, test_y, avg=False)
    return self.mean_r_scores

