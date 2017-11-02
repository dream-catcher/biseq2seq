#!/usr/bin/env python3
import os
import sys

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
            q_line = q_line.split(SEP)[1]
            r_line = r_line.split(SEP)[1]
            a_line = a_line.split(SEP)[1]
    
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


