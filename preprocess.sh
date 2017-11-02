#sudo cp -frv /media/sf_VM_SHARE/fin_qa_corpus ~/workspace/data/fin_qa/
#
#sudo chown -R emerson:emerson ~/workspace/data/fin_qa/fin_qa_corpus
#
BASE_DIR=/home/emerson/workspace/data/fin_qa/fin_qa_corpus_20170913
pushd dataset/fin_qa/

./clean_corpus.py ${BASE_DIR}

echo "start to process words mapping"
./map_words.py map ${BASE_DIR}/questions_out.txt ${BASE_DIR}/questions_preprocessed.txt
./map_words.py map ${BASE_DIR}/orig_answer_out.txt ${BASE_DIR}/orig_answer_preprocessed.txt
./map_words.py map ${BASE_DIR}/retrieval_answer_out.txt ${BASE_DIR}/retrieval_answer_preprocessed.txt

#
python3 partition_dataset.py
#
## generate vocabulary
cd train
#echo "generate vocabulary"
python3 ../../../bin/tools/generate_vocab.py sources_query.txt > vocab.sources.txt
python3 ../../../bin/tools/generate_vocab.py sources_candidate.txt > vocab.sources_candidate.txt
python3 ../../../bin/tools/generate_vocab.py targets.txt > vocab.targets.txt

popd

