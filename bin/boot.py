#! /usr/bin/env python
import os
import subprocess

BASE_DIR = "dataset/chitchat"
VOCAB_SOURCE = os.path.join(BASE_DIR, "train/vocab.sources.txt")
VOCAB_TARGET = os.path.join(BASE_DIR, "train/vocab.targets.txt")
TRAIN_SOURCES_QUERY = os.path.join(BASE_DIR, "train/sources_query.txt")
TRAIN_SOURCES_CANDIDATE = os.path.join(BASE_DIR, "train/sources_candidate.txt")
TRAIN_TARGETS = os.path.join(BASE_DIR, "train/targets.txt")
DEV_SOURCES_QUERY = os.path.join(BASE_DIR, "dev/sources_query.txt")
DEV_SOURCES_CANDIDATE = os.path.join(BASE_DIR, "dev/sources_candidate.txt")
DEV_TARGETS = os.path.join(BASE_DIR, "dev/targets.txt")

DEV_TARGETS_REF = os.path.join(BASE_DIR, "dev/targets.txt")
TRAIN_STEPS = 1000

MODEL_DIR = os.path.join("/tmp", "chitchat_bis2s")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

os.chdir("..")

os.system(
    '''python -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: dataset/chitchat/train/vocab.sources.txt
      vocab_source_candidate: dataset/chitchat/train/vocab.sources_candidate.txt
      vocab_target: dataset/chitchat/train/vocab.targets.txt" \
  --input_pipeline_train "
    class: TripleTextInputPipeline
    params:
      query_files:
        - dataset/chitchat/train/sources_query.txt
      candidate_files:
        - dataset/chitchat/train/sources_candidate.txt
      target_files:
        - dataset/chitchat/train/targets.txt" \
  --input_pipeline_dev "
    class: TripleTextInputPipeline
    params:
       query_files:
        - dataset/chitchat/dev/sources_query.txt
       candidate_files:
        - dataset/chitchat/dev/sources_candidate.txt
       target_files:
        - dataset/chitchat/dev/targets.txt" \
  --batch_size 32 \
  --train_steps 1000 \
  --output_dir chitchat_bis2s''')