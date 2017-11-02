import os,sys

BASE_DIR=/home/emerson/workspace/data/fin_qa/fin_qa_corpus_20170913

pushd dataset/fin_qa/
FileTuple = namedtuple("FileTuple", ["query", "candidate", "target"]))

def work_flow(work_dir, corpus_dir):
    cur_dir = os.path.abspath(".")
    os.chdir(work_dir)
    clean_cmd = "./clean_corpus.py {}".format(corpus_dir)
    os.system(clean_cmd)
    os.chdir(cur_dir)


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

#!/usr/bin/env python3
import os
import sys

fin_list = FileTuple("questions.txt", "retrieval_answer.txt", "orig_answer.txt")
fout_list = FileTuple("questions.txt", "retrieval_answer.txt", "orig_answer.txt")

FIN_QUESTION = "questions.txt"

FIN_RETRIEVAL = "retrieval_answer.txt"

FIN_ANSWER = "orig_answer.txt"

FOUT_QUESTION = "questions_out.txt"

FOUT_RETRIEVAL = "retrieval_answer_out.txt"

FOUT_ANSWER = "orig_answer_out.txt"

SEP = "#SEP#" 
def clean(work_dir):
    last = None
    os.chdir(work_dir)
    cnt = 0
    with open(FIN_QUESTION) as fi_q, open(FIN_RETRIEVAL) as fi_r, open(FIN_ANSWER) as fi_a,\
        open(FOUT_QUESTION, "w") as fo_q, open(FOUT_RETRIEVAL, "w") as fo_r, open(FOUT_ANSWER, "w") as fo_a:
        for q_line in fi_q:
            r_line = fi_r.readline()
            a_line = fi_a.readline()
            q_idx,q_line = q_line.split(SEP)
            r_idx,r_line = r_line.split(SEP)
            a_idx,a_line = a_line.split(SEP)
    
            assert (q_idx == r_idx == a_idx)
            tag = q_line + r_line + a_line
            if last == tag:
                print("ignore duplicated lines")
                continue
            
            cnt += 1
            fo_q.write(q_line)
            fo_r.write(r_line)
            fo_a.write(a_line)
            last = tag


    print("total lines:{}".format(cnt))


if __name__ == "__main__":
    work_dir = "fin_qa_corpus"
    if len(sys.argv) > 1:
        work_dir = sys.argv[1]
    print("start to clean in directory:{}".format(work_dir))
    clean(work_dir)
    print("finish cleaning")


