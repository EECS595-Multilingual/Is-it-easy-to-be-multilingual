from __future__ import absolute_import, division, print_function

import argparse
from transformers import BertTokenizer, XLMTokenizer, XLMRobertaTokenizer
import os
from collections import defaultdict
import csv
import random
import os
import shutil
import json


TOKENIZERS = {
  'bert': BertTokenizer,
  'xlm': XLMTokenizer,
  'xlmr': XLMRobertaTokenizer,
}

def panx_preprocess(args):
  def _process_one_file(infile, outfile):
    with open(infile, 'r+') as fin, open(outfile, 'w+') as fout:
      for l in fin:
        items = l.strip().split('\t')
        if len(items) == 2:
          label = items[1].strip()
          token = items[0].split(':')[1].strip()
          if 'test' in infile:
            fout.write(f'{token}\n')
          else:
            fout.write(f'{token}\t{label}\n')
        else:
          fout.write('\n')
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)
  langs = 'en fi ar bn id ko ru sw te'.split(' ')
  for lg in langs:
    for split in ['train', 'test', 'dev']:
      infile = os.path.join(args.data_dir, f'{lg}-{split}')
      outfile = os.path.join(args.output_dir, f'{split}-{lg}.tsv')
      _process_one_file(infile, outfile)

def tydiqa_preprocess(args):
  LANG2ISO = {'arabic': 'ar', 'bengali': 'bn', 'english': 'en', 'finnish': 'fi',
              'indonesian': 'id', 'korean': 'ko', 'russian': 'ru',
              'swahili': 'sw', 'telugu': 'te'}
  assert os.path.exists(args.data_dir)
  train_file = os.path.join(args.data_dir, 'tydiqa-goldp-v1.1-train.json')
  os.makedirs(args.output_dir, exist_ok=True)

  # Split the training file into language-specific files
  lang2data = defaultdict(list)
  with open(train_file, 'r') as f_in:
    data = json.load(f_in)
    version = data['version']
    for doc in data['data']:
      for par in doc['paragraphs']:
        context = par['context']
        for qa in par['qas']:
          question = qa['question']
          question_id = qa['id']
          example_lang = question_id.split('-')[0]
          q_id = question_id.split('-')[-1]
          for answer in qa['answers']:
            a_start, a_text = answer['answer_start'], answer['text']
            a_end = a_start + len(a_text)
            assert context[a_start:a_end] == a_text
          lang2data[example_lang].append({'paragraphs': [{
              'context': context,
              'qas': [{'answers': qa['answers'],
                       'question': question,
                       'id': q_id}]}]})

  for lang, data in lang2data.items():
    out_file = os.path.join(
        args.output_dir, 'tydiqa.%s.train.json' % LANG2ISO[lang])
    with open(out_file, 'w') as f:
      json.dump({'data': data, 'version': version}, f)

  # Rename the dev files
  dev_dir = os.path.join(args.data_dir, 'tydiqa-goldp-v1.1-dev')
  assert os.path.exists(dev_dir)
  for lang, iso in LANG2ISO.items():
    src_file = os.path.join(dev_dir, 'tydiqa-goldp-dev-%s.json' % lang)
    dst_file = os.path.join(dev_dir, 'tydiqa.%s.dev.json' % iso)
    os.rename(src_file, dst_file)

  # Remove the test annotations to prevent accidental cheating
  remove_qa_test_annotations(dev_dir)


def remove_qa_test_annotations(test_dir):
  assert os.path.exists(test_dir)
  for file_name in os.listdir(test_dir):
    new_data = []
    test_file = os.path.join(test_dir, file_name)
    with open(test_file, 'r') as f:
      data = json.load(f)
      version = data['version']
      for doc in data['data']:
        for par in doc['paragraphs']:
          context = par['context']
          for qa in par['qas']:
            question = qa['question']
            question_id = qa['id']
            for answer in qa['answers']:
              a_start, a_text = answer['answer_start'], answer['text']
              a_end = a_start + len(a_text)
              assert context[a_start:a_end] == a_text
            new_data.append({'paragraphs': [{
                'context': context,
                'qas': [{'answers': [{'answer_start': answer['answer_start'], 'text': answer['text']}],
                         'question': question,
                         'id': question_id}]}]})
    with open(test_file, 'w') as f:
      json.dump({'data': new_data, 'version': version}, f)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--data_dir", default=None, type=str, required=True,
                      help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--output_dir", default=None, type=str, required=True,
                      help="The output data dir where any processed files will be written to.")
  parser.add_argument("--task", default="panx", type=str, required=True,
                      help="The task name")
  parser.add_argument("--model_name_or_path", default="bert-base-multilingual-cased", type=str,
                      help="The pre-trained model")
  parser.add_argument("--model_type", default="bert", type=str,
                      help="model type")
  parser.add_argument("--max_len", default=512, type=int,
                      help="the maximum length of sentences")
  parser.add_argument("--do_lower_case", action='store_true',
                      help="whether to do lower case")
  parser.add_argument("--cache_dir", default=None, type=str,
                      help="cache directory")
  parser.add_argument("--languages", default="en", type=str,
                      help="process language")
  parser.add_argument("--remove_last_token", action='store_true',
                      help="whether to remove the last token")
  parser.add_argument("--remove_test_label", action='store_true',
                      help="whether to remove test set label")
  args = parser.parse_args()


  if args.task == 'tydiqa':
    tydiqa_preprocess(args)
  if args.task == 'panx':
    panx_preprocess(args)