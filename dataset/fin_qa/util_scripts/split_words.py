#!/usr/bin/env python3

import sys
import jieba

def split(fi_name, fo_name):
    with open(fi_name) as fi, open(fo_name, "w") as fo:
        for l in fi:
            words = [w for w in jieba.cut(l.strip())]
            line = ' '.join(words)
            fo.write(line + "\n")

           
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("format: split.py in_file out_file")
        exit()
    jieba.load_userdict("user.dict")
    fin = sys.argv[1]
    fout = sys.argv[2]
    split(fin, fout)
    print("finish")
