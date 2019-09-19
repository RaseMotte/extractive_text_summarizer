import torch

def _make_single_string(text_tokens_list):
    """
    Make text into two single strings, putting <s> and </s> tags around the sentences.
    """
    text_sent_list = [' '.join(token_list)
                         for token_list in text_tokens_list]
    text = ' '.join(sent for sent in text_sent_list)
    return text

def load_chunk_corpus(chunk_file):
  chunk = torch.load(chunk_file)
  hash_list = []
  article_list = []
  summary_list = []
  for story in chunk:
    article_tokens_list = story["article"]
    if len(article_tokens_list) <= 1:
      continue
    oracle_ids = story["oracle_ids"]
    summary_tokens_list = [article_tokens_list[oid] for oid in oracle_ids]
    article = _make_single_string(article_tokens_list)
    summary = _make_single_string(summary_tokens_list)
    hash_list.append(story["url_hash"])
    article_list.append(article)
    summary_list.append(summary)
  return hash_list, article_list, summary_list