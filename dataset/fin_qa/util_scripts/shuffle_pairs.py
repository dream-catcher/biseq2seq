#!/usr/bin/env python
import numpy as np
import sys

SRC_FILE = "question.txt"
SRC_SHUFFLE_FILE = "shuffle_question.txt"
TARGET_FILE = "answer.txt"
TARGET_SHUFFLE_FILE = "shuffle_answer.txt"

def shuffle(fin1_name, fout1_name, fin2_name, fout2_name):
    print("start!")
    lines1 = open(fin1_name).readlines()
    lines2 = open(fin2_name).readlines()
    print("shuffle indices")
    shuffle_indices = np.random.permutation(len(lines1))
    fout1 = open(fout1_name, "w")
    fout2 = open(fout2_name, "w")
    print("output lines:")
    for i in shuffle_indices:
        fout1.write(lines1[i])
        fout2.write(lines2[i])
    print("finish!")


if __name__ == "__main__":

    print("start to shuffle")
    shuffle(SRC_FILE, SRC_SHUFFLE_FILE, TARGET_FILE, TARGET_SHUFFLE_FILE)
    print("finish shuffling")


