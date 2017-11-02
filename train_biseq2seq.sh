export BASE_DIR=dataset/chitchat
export VOCAB_SOURCE=${BASE_DIR}/train/vocab.sources.txt
export VOCAB_SOURCE_CANDIDATE=${BASE_DIR}/train/vocab.sources_candidate.txt
export VOCAB_TARGET=${BASE_DIR}/train/vocab.targets.txt
export TRAIN_SOURCES_QUERY=${BASE_DIR}/train/sources_query.txt
export TRAIN_SOURCES_CANDIDATE=${BASE_DIR}/train/sources_candidate.txt
export TRAIN_TARGETS=${BASE_DIR}/train/targets.txt
export DEV_SOURCES_QUERY=${BASE_DIR}/dev/sources_query.txt
export DEV_SOURCES_CANDIDATE=${BASE_DIR}/dev/sources_candidate.txt
export DEV_TARGETS=${BASE_DIR}/dev/targets.txt

export DEV_TARGETS_REF=${BASE_DIR}/dev/targets.txt
export TRAIN_STEPS=1000

export MODEL_DIR=${TMPDIR:-/tmp}/chitchat_bis2s
mkdir -p $MODEL_DIR

python3 -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_source_candidate: $VOCAB_SOURCE_CANDIDATE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: TripleTextInputPipeline
    params:
      query_files:
        - $TRAIN_SOURCES_QUERY
      candidate_files:
        - $TRAIN_SOURCES_CANDIDATE
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: TripleTextInputPipeline
    params:
       query_files:
        - $DEV_SOURCES_QUERY
       candidate_files:
        - $DEV_SOURCES_CANDIDATE
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR

