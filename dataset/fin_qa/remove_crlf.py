# -*- coding: UTF-8 -*-
# date: 2017-9-13
# author: gaoyi

import os

FIN_NAME = "d:/temp/merge_folder/retrieval_answer.txt"
FOUT_NAME = "d:/temp/merge_folder/retrieval_answer_out.txt"
PARTS_NUM = 2
SEP = "#SEP#"

last_label = None
session = []
with open(FIN_NAME) as fi, open(FOUT_NAME, "w") as fo:
	for cnt,l in enumerate(fi):
		if cnt % 1000 == 0:
			print("processing lines:{}".format(cnt))
		line = l.strip()
		parts = line.split(SEP)
		if len(parts) != PARTS_NUM:
			print("format error line:{}".format(cnt))
			session.append(line)
		else:
			# 1.write out last session first if need
			if session and last_label:
				fo.write(last_label + SEP + ''.join(session) + "\n")
				last_label = None
				session = []
			# 2.save current line information
			label,contents = parts
			if last_label != label:
				last_label = label
			session.append(contents)
	if session and last_label:
		fo.write(last_label + SEP + ''.join(session) + "\n")


