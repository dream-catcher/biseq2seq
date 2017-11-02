#!/usr/bin/env python3

import re


pattern = r"\$\{([A-Z_]+)\}"

FIN = "a.txt"
FOUT = "a_new.txt"

def replace(fin, fout):
    symbols = set()
    with open(fin) as fi, open(fout, "w") as fo:
        for l in fi:
            result = re.findall(pattern, l.strip())
            line = re.sub(pattern, r"#\1#", l.strip())
            fo.write(line + "\n")
            if result:
                s = result[0]
                symbols.add(s)
    return symbols


if __name__ == "__main__":
    symbols = replace(FIN, FOUT)
    print("-------------------------")
    with open("special_tokens.txt", "w") as f_t:
        for s in symbols:
            print(s)
            f_t.write("#" + s + "# 100\n")



