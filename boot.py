#! /usr/bin/env python
import os
import sys
import subprocess
import argparse

#BASE_DIR = "dataset/fin_qa"
#VOCAB_SOURCE = os.path.join(BASE_DIR, "train/vocab.sources.txt")
#VOCAB_TARGET = os.path.join(BASE_DIR, "train/vocab.targets.txt")
#TRAIN_SOURCES_QUERY = os.path.join(BASE_DIR, "train/sources_query.txt")
#TRAIN_SOURCES_CANDIDATE = os.path.join(BASE_DIR, "train/sources_candidate.txt")
#TRAIN_TARGETS = os.path.join(BASE_DIR, "train/targets.txt")
#DEV_SOURCES_QUERY = os.path.join(BASE_DIR, "dev/sources_query.txt")
#DEV_SOURCES_CANDIDATE = os.path.join(BASE_DIR, "dev/sources_candidate.txt")
#DEV_TARGETS = os.path.join(BASE_DIR, "dev/targets.txt")
#
#DEV_TARGETS_REF = os.path.join(BASE_DIR, "dev/targets.txt")
#TRAIN_STEPS = 50000
#
#MODEL_DIR = os.path.join(".", "model_fin_qa")
#if not os.path.exists(MODEL_DIR):
#    os.makedirs(MODEL_DIR)

parameters = ""
parser = argparse.ArgumentParser()
parser.add_argument("--model_output_path", default="", help="model saved path", type=str)
args = sys.argv
result, unparsed = parser.parse_known_args(args=args)
print("arguments:{}".format(result))
if "model_output_path" in result:
    parameters = "--model_output_path={}".format(result.model_output_path)

os.system(
    '''python2 -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: dataset/fin_qa/train/vocab.sources.txt
      vocab_source_candidate: dataset/fin_qa/train/vocab.sources_candidate.txt
      vocab_target: dataset/fin_qa/train/vocab.targets.txt" \
  --input_pipeline_train "
    class: TripleTextInputPipeline
    params:
      query_files:
        - dataset/fin_qa/train/sources_query.txt
      candidate_files:
        - dataset/fin_qa/train/sources_candidate.txt
      target_files:
        - dataset/fin_qa/train/targets.txt" \
  --input_pipeline_dev "
    class: TripleTextInputPipeline
    params:
       query_files:
        - dataset/fin_qa/dev/sources_query.txt
       candidate_files:
        - dataset/fin_qa/dev/sources_candidate.txt
       target_files:
        - dataset/fin_qa/dev/targets.txt" \
  --batch_size 64 \
  --train_steps 300000 \
  --output_dir model_fin_qa ''' + parameters)


