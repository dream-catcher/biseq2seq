#!/usr/bin/env python3
import os
import re
from collections import Counter

BASE_DIR = "/home/"
FILE_LIST = "orig_answer.txt  questions.txt  retrieval_answer.txt".split()
pattern = u"【.*?】"

SYMBOL_PREFIX = "ENTITY_"

def find_entity(file_list, fout):
  words = []
  for f in file_list:
    with open(os.path.join(BASE_DIR, f)) as fi:
      for l in fi:
        result = re.findall(pattern, l)
        if result:
          for r in result:
            print(r)
            words.append(r)
  counter = Counter(words)
  with open(fout, "w") as fo:
    number = 0
    for k,v in counter.items():
      number += 1
      fo.write(k + "\t" + SYMBOL_PREFIX + str(number) + "\t" + str(v) + "\n")
   



if __name__ == "__main__":
    flist = FILE_LIST
    find_entity(flist, "entity.txt")
