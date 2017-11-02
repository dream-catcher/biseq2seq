#!/usr/bin/env python3

import os
import re
import sys
import json

vocab = json.load(open("vocab.json"))
reverse_vocab = {v:k for k,v in vocab.items()}
REMOVE_TAG = "#REMOVE#"

def restore(fin_name, fout_name):
    with open(fin_name) as fi, open(fout_name, "w") as fo:
        for line_cnt,line in enumerate(fi):
            words = line.strip().split()
            #if line_cnt > 1000:
            #    break
            for i,w in enumerate(words):
                if w in reverse_vocab:
                    if re.match("\d+", reverse_vocab[w]):
                        words[i] = reverse_vocab[w]
                        print("{} line:{} stage1 replace {} --> {}".format(fin_name, line_cnt, w, reverse_vocab[w]))

            line = ' '.join(words)
            print("replace numbers line:" + line)
            words = line.strip().split()
            for i,w in enumerate(words):
                if words[i].startswith("SYMBOL") and i < len(words) - 1 and re.match("\d+", words[i+1]):
                    words[i] = words[i] + words[i+1]
                    print("{} line:{} stage2 merge:{} {}".format(fin_name, line_cnt, words[i], words[i+1])) 
                    words[i+1] = REMOVE_TAG
            words = [w for w in words if w != REMOVE_TAG]
            line = ' '.join(words)
            print("after merged line:" + line)

            words = line.split()
            for i,w in enumerate(words):
                if w in reverse_vocab:
                    print("{} line:{} stage3 replace {} --> {}".format(fin_name, line_cnt, w, reverse_vocab[w]))
                    words[i] = reverse_vocab[w]
            fo.write(' '.join(words) + "\n")
            print("final:" + ''.join(words) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("format: restore.py infile outfile")
        exit(-1)
    fin_name = sys.argv[1]
    fout_name = sys.argv[2]
    print("start to process")
    restore(fin_name, fout_name)
    print("finish")


