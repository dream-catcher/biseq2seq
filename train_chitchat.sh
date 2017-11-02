export BASE_DIR=dataset/chitchat
export VOCAB_SOURCE=${BASE_DIR}/train/vocab.sources.txt
export VOCAB_TARGET=${BASE_DIR}/train/vocab.targets.txt
export TRAIN_SOURCES=${BASE_DIR}/train/sources.txt
export TRAIN_TARGETS=${BASE_DIR}/train/targets.txt
export DEV_SOURCES=${BASE_DIR}/dev/sources.txt
export DEV_TARGETS=${BASE_DIR}/dev/targets.txt

export DEV_TARGETS_REF=${BASE_DIR}/dev/targets.txt
export TRAIN_STEPS=1000

export MODEL_DIR=${TMPDIR:-/tmp}/chitchat_s2s
mkdir -p $MODEL_DIR

python3 -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR

