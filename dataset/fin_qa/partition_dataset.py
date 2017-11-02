#!/usr/bin/env python3

import os
import re
#import jieba
import json
import numpy as np

BASE_DIR = "/home/emerson/workspace/data/fin_qa/fin_qa_corpus_20170913"
#BASE_DIR = "fin_qa_corpus"
SOURCE_FILE = os.path.join(BASE_DIR, "questions_split.txt")
TARGET_FILE = os.path.join(BASE_DIR, "orig_answer_split.txt")
CANDIDATE_FILE = os.path.join(BASE_DIR, "retrieval_answer_split.txt")

RATIO = 0.9
TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
SOURCE_NAME = "sources_query.txt"
CANDIDATE_NAME = "sources_candidate.txt"
TARGET_NAME = "targets.txt"


#PATTERN_STR_LIST = [r"[-:\d]{3,}|\d+\.\d+",
#                r"\$\s*\{.*?(?:\}|｝)",
#                r"【.*?】"]
#PATTERN_LIST = [re.compile(p) for p in PATTERN_STR_LIST]
#SYMBOL_FMT = "SYMBOL{}"
#REMOVE_WORD = '#REMOVE#'
#
#vocab = {}
#symbol_cnt = 0

#def split_words(input_line):
#    words = jieba.lcut(line)
#    return ' '.join(words)
    ## 1.replace special tokens with tags
    #global vacab
    #global symbol_cnt
    #line = input_line.strip()
    #for p in PATTERN_LIST:
    #    result = p.findall(line)
    #    local_vacab = {}
    #    #print("result:{}".format(result)); keys = [r for r in result if r not in keys] ; print("result:{}".format(result))
    #    result = list(set(result))
    #    for i,r in enumerate(result):
    #        if r not in vocab:
    #            symbol_cnt += 1
    #            symbol = SYMBOL_FMT.format(symbol_cnt)
    #            vocab[r] = symbol
    #            local_vocab[r] = symbol

    #    for k,v in local_vocab.items():
    #        line = line.replace(k, " " + v + " ")
    #    
    #words = jieba.lcut(line)
    #return ' '.join(words)
    ##print(words)
    #reverse_vocab = {v:k for k,v in vocab.items()}
    ##print("reverse_vocab:{}".format(reverse_vocab))
    ## 2.restore special tokens
    #for i, w in enumerate(words):
    #    if w in reverse_vocab:
    #        words[i] = reverse_vocab[w]
    #        if i-1 >= 0 and words[i-1] == " ":
    #            words[i-1] = REMOVE_WORD
    #        if i+1 < len(words) and words[i+1] == " ":
    #            words[i+1] = REMOVE_WORD
    #words = [w for w in words if w != REMOVE_WORD]
    ##print(words)
    ##return ' '.join(words)
    #ret = ' '.join(words)
    ##print("line split:\n{}-->\n{}".format(input_line, ret))
    #return ret

def write_dataset(dataset, indices, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    source_file = os.path.join(folder_name, SOURCE_NAME)
    target_file = os.path.join(folder_name, TARGET_NAME)
    candidate_file = os.path.join(folder_name, CANDIDATE_NAME)
    q_lines, a_lines, r_lines = dataset
    print("writing dataset :{}".format(folder_name))
    cnt = 0
    with open(source_file, "w") as fs, open(target_file, "w") as ft, open(candidate_file, "w") as fr:
        for i in indices:
            cnt += 1
            #ql = q_lines[i]
            #al = a_lines[i]
            #print("ql:{} al:{}".format(ql, al))
            #q_words = split_words(ql)
            #a_words = split_words(al)
            #print("q_words:{} a_words:{}".format(q_words, a_words))
            fs.write(q_lines[i])
            ft.write(a_lines[i])
            fr.write(r_lines[i])
            if cnt % 1000 == 1:
                print("{} processing lines:{}".format(folder_name, cnt))

def partition():
    with open(SOURCE_FILE) as fq, open(TARGET_FILE) as fa, open(CANDIDATE_FILE) as fr:
        # 1.read all lines in memory
        print("# 1.read all lines in memory")
        #num_lines = sum(1 for line in fq)
        q_lines = fq.readlines()
        a_lines = fa.readlines()
        r_lines = fr.readlines()
        assert len(q_lines) == len(a_lines)
        assert len(q_lines) == len(r_lines)
        total = len(q_lines)

        # 2.shuffle q/a pairs
        print("# 2.shuffle q/a pairs")
        shuffle_indices = np.random.permutation(total)

        # 3.partition according to ratio
        print("# 3.partition according to ratio")
        train_len = int(total * RATIO)
        print("split train/test/dev: {} / {} / {}".format(train_len, total-train_len-10, 10))
        dataset = [q_lines, a_lines, r_lines]
        train_set_indices = shuffle_indices[:train_len]
        test_set_indices = shuffle_indices[train_len:-10]
        dev_set_indices = shuffle_indices[-10:]
        print("train_set_indices:{} ".format(dev_set_indices))
        #train_set = [q_lines[:train_len], a_lines[:train_len]]
        #test_set = [q_lines[train_len:-10], a_lines[train_len:-10]]
        #dev_set = [q_lines[-10:], a_lines[-10:]]

        # 4.write train set
        print("# 4.write train set")
        write_dataset(dataset, train_set_indices, TRAIN_SET)
        write_dataset(dataset, test_set_indices, TEST_SET)
        write_dataset(dataset, dev_set_indices, DEV_SET)

if __name__ == "__main__":
    print("start to process:")
    #jieba.load_userdict("user.dict")
    partition()
    #json.dump(vocab, open("vocab.json", "w"))
    print("finish")
