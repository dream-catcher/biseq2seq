#!/usr/bin/env python3

import os
import re


FILE_LIST = ["q_2017-08-17.txt", "a_2017-08-17.txt"]
pattern = u"\$\{(.*?)\}"

def find_tokens(file_list, fout):
  for f in file_list:
    with open(f) as fi, open(fout, "w") as fo:
      for l in fi:
        result = re.findall(pattern, l)
        if result:
          for r in result:
            #s = re.sub(pattern, r"\1", r)
            #print("{}-->{}".format(r, s))
            fo.write("#{}#\n".format(r))

if __name__ == "__main__":
  print("start")
  find_tokens(FILE_LIST, "tokens.txt")
  print("generating tokens vocab")
  os.system("python3 ../../bin/tools/generate_vocab.py tokens.txt > sorted_tokens.txt")
  print("finish")
