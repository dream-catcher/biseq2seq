# -*- coding: UTF-8 -*-
import os 

F_QUESTION = "sources_query.txt"
F_ANSWER = "targets.txt"
F_RETRIEVAL = "sources_candidate.txt"
F_PREDICTION = "predictions.txt"
F_MERGE = "merge.txt"

os.chdir("test")
with open(F_QUESTION) as fq, open(F_ANSWER) as fa, open(F_RETRIEVAL) as fr, \
    open(F_PREDICTION) as fp, open(F_MERGE, "w") as fo:
    for q_line in fq:
        a_line = fa.readline()
        r_line = fr.readline()
        p_line = fp.readline()
        fo.write("--------------------------\n")
        fo.write("q:" + q_line)
        fo.write("a:" + a_line)
        fo.write("r:" + r_line)
        fo.write("p:" + p_line)
