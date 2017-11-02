export BASE_DIR=dataset/fin_qa
export TEST_SOURCES_QUERY=${BASE_DIR}/test/sources_query.txt
export TEST_SOURCES_CANDIDATE=${BASE_DIR}/test/sources_candidate.txt
export TEST_TARGETS=${BASE_DIR}/test/targets.txt

export MODEL_DIR=model_fin_qa

export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5" \
  --input_pipeline "
    class: TripleTextInputPipeline
    params:
      query_files:
        - $TEST_SOURCES_QUERY
      candidate_files:
        - $TEST_SOURCES_CANDIDATE" \
  | tee ${PRED_DIR}/predictions.txt
