from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import unicodedata


PUNCT = {chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')}.union(string.punctuation)
WHITESPACE_LANGS = ['de']


def whitespace_tokenize(text):
    return text.split()


def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r'[\u4e00-\u9fa5]', char) or char in PUNCT:
            if temp_str != "":
                ss = whitespace_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b', ' ', text)

    def white_space_fix(text):
        tokens = whitespace_tokenize(text)
        return ' '.join([t for t in tokens if t.strip() != ''])

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

with open('/content/drive/MyDrive/xlmr_viquad/GermanQuAD/GermanQuAD_test.json') as f:
  data = json.load(f)['data']

with open('/content/drive/MyDrive/xlmr_viquad/xlmr-finetuned-GermanQuAD/predictions_eval.json') as f:
  predictions = json.load(f)

f1_sum, em_sum =0.0, 0.0
dem = 0 
for a in data:
  for p in a['paragraphs']:
    for qa in p['qas']:
      dem += 1
      prediction = predictions[str(qa['id'])]
      answers = [ans['text'] for ans in qa['answers']]
      f1, em =0, 0
      for answer in answers:
        f1 = max(f1_score(prediction, answer), f1)
        em = max(exact_match_score(prediction, answer), em)
      f1_sum += f1
      em_sum += em

print("Number of sample: ", dem)
print("F1-score: ", f1_sum/dem)
print("Exact Match: ", em_sum/dem)

