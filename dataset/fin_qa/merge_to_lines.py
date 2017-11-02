# -*- coding: UTF-8 -*-
import os 

F_SOURCE_QUERY = "sources_query_r.txt"
F_SOURCE_CANDIDATE = "sources_candidate_r.txt"
F_ANSWER = "targets_r.txt"
F_PREDICTION = "predictions_r.txt"
F_MERGE = "merge_xlsx.txt"

os.chdir("test")
with open(F_SOURCE_QUERY) as fsq, open(F_SOURCE_CANDIDATE) as fsc, open(F_ANSWER) as fa, \
    open(F_PREDICTION) as fp, open(F_MERGE, "w") as fo:
    for q_line in fsq:
        qc_line = fsc.readline()
        a_line = fa.readline()
        p_line = fp.readline()
        q = ''.join(q_line.strip().split())
        qc = ''.join(qc_line.strip().split())
        a = ''.join(a_line.strip().split())
        p = ''.join(p_line.strip().split())
        fo.write("{}\t{}\t{}\t{}\r\n".format(q, qc, a, p))
