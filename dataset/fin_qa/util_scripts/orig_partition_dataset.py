#!/usr/bin/env python3

import os
import re
import jieba
import numpy as np

SOURCE_FILE = "q_2017-08-17.txt"
TARGET_FILE = "a_2017-08-17.txt"
RATIO = 0.9
TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
SOURCE_NAME = "sources.txt"
TARGET_NAME = "targets.txt"


pattern = r"\$\s*\{(.*?)(?:\}|｝)"

def split_words(line):
    line = re.sub(pattern, r"#\1#", line.strip())
    return ' '.join([w for w in jieba.cut(line.strip())])

def write_dataset(dataset, indices, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    source_file = os.path.join(folder_name, SOURCE_NAME)
    target_file = os.path.join(folder_name, TARGET_NAME)
    q_lines, a_lines = dataset
    with open(source_file, "w") as fs, open(target_file, "w") as ft:
        for i in indices:
            #ql = q_lines[i]
            #al = a_lines[i]
            #print("ql:{} al:{}".format(ql, al))
            #q_words = split_words(ql)
            #a_words = split_words(al)
            #print("q_words:{} a_words:{}".format(q_words, a_words))
            fs.write(split_words(q_lines[i]) + "\n")
            ft.write(split_words(a_lines[i]) + "\n")

def partition():
    with open(SOURCE_FILE) as fq, open(TARGET_FILE) as fa:
        # 1.read all lines in memory
        print("# 1.read all lines in memory")
        #num_lines = sum(1 for line in fq)
        q_lines = fq.readlines()
        a_lines = fa.readlines()
        assert len(q_lines) == len(a_lines)
        total = len(q_lines)

        # 2.shuffle q/a pairs
        print("# 2.shuffle q/a pairs")
        shuffle_indices = np.random.permutation(total)

        # 3.partition according to ratio
        print("# 3.partition according to ratio")
        train_len = int(total * RATIO)
        print("split train/test/dev: {} / {} / {}".format(train_len, total-train_len-10, 10))
        dataset = [q_lines, a_lines]
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
    jieba.load_userdict("user.dict")
    partition()
    print("finish")
