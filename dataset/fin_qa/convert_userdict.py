FI = "fin_vocab.txt"
FO = "fin_vocab_dict.txt"
with open(FI) as fi, open(FO, "w") as fo:
    for s in fi:
        s = s.strip()
        print(s)
        fo.write(s + " 100\n")
