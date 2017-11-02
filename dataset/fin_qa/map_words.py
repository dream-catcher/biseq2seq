#!/usr/bin/env python3
import os
import re
import sys
import json
import jieba

PATTERN_STR_LIST = [r"\$\s*\{.*?(?:\}|｝)",
                    r"【.*?】",
                    r"[-:%\d]{3,}|\d+\.\d+"]

PATTERN_LIST = [re.compile(p) for p in PATTERN_STR_LIST]

SYMBOL_FMT = "SYMBOL{}"

REMOVE_WORD = "#REMOVE#"

VOCAB_FILE = "special_symbols.json"

class StringPiece(object):
    def __init__(self, string, is_orig=True):
        self.string = string
        self.is_orig = is_orig
    def __repr__(self):
        return "[string={}, is_orig={}]".format(self.string, self.is_orig)

def map_words(input_line, vocab):
    line = input_line.strip()
    piece_list = [StringPiece(line)]
    new_list = []
    for p in PATTERN_LIST:
        for piece in piece_list:
            #print("map_words piece:{} for p:{}".format(piece, p)) 
            if piece.is_orig:
                new_list += search_all(piece.string, p, vocab)
            else:
                new_list += [piece]
        piece_list = new_list
        new_list = []
    final_string = ' '.join([p.string for p in piece_list])
    #print(final_string)
    cut_list = jieba.lcut(final_string)
    #print("cut_list:{}".format(cut_list))
    words = clean_words_list(cut_list)
    #print("clean words:{}".format(words))
    r = ' '.join(words)
    return r

def clean_words_list(words):
    total = len(words)
    for i,w in enumerate(words):
        if w.startswith("SYMBOL"):
            if i-1 > 0 and words[i-1] == ' ':
                words[i-1] = REMOVE_WORD
            if i+1 < total and words[i+1] == ' ':
                words[i+1] = REMOVE_WORD
    ret_words = [w for w in words if w != REMOVE_WORD]
    return ret_words

def search_all(line, pattern, vocab):
    piece_list = []
    result = pattern.search(line)
    last_end = 0
    total = len(line)
    while result:
        start = result.start()
        end = result.end()
        if start > last_end:
            piece_list.append(StringPiece(line[last_end:start]))
        w = result.group(0)
        if w not in vocab:
            symbol = SYMBOL_FMT.format(len(vocab))
            vocab[w] = symbol
        else:
            symbol = vocab[w]
        piece_list.append(StringPiece("{}".format(symbol), False))
        last_end = end
        if end < total:
            result = pattern.search(line, end)
        else:
            break
    if last_end < total:
       piece_list += [StringPiece(line[last_end:])]
    #print("search_all for line:{} for pattern:{} return:{}".format(line, pattern, piece_list))
    return piece_list

def restore_words(input_line, reverse_vocab):
    print("input_line:{}".format(input_line))
    words = input_line.strip().split(" ")
    for i,w in enumerate(words):
        if w in reverse_vocab:
            words[i] = reverse_vocab[w]
    print("restore words:{}".format(words))
    for i,w in enumerate(words):
        if w == '':
            words[i] = ' '
    print("processed words:{}".format(words))
    return ''.join(words)


if __name__ == "__main__":
    jieba.load_userdict("user.dict")
    if len(sys.argv) != 4:
        print("format: map_words.py cmd infile outfile")
        print("cmd:map / restore")
        exit(-1)

    cmd = sys.argv[1]
    assert cmd in ('map', 'restore', 'test')
    fin_name = sys.argv[2]
    fout_name = sys.argv[3]

    #---------------------------
    if cmd == 'map':
        if os.path.exists(VOCAB_FILE):
            vocab = json.load(open(VOCAB_FILE))
        else:
            vocab = {}
        with open(fin_name) as fi, open(fout_name, "w") as fo:
            for cnt,line in enumerate(fi):
                if cnt % 1000 == 0:
                    print("processing cnt:{}".format(cnt))
                l = map_words(line, vocab)
                fo.write(l + "\n")
        # save vocab
        json.dump(vocab, open(VOCAB_FILE, "w"))

    elif cmd == 'restore':
        vocab = json.load(open(VOCAB_FILE))
        reverse_vocab = {v:k for k,v in vocab.items()}
        with open(fin_name) as fi, open(fout_name, "w") as fo:
            for line in fi:
                l = restore_words(line, reverse_vocab)
                fo.write(l + "\n")
    elif cmd == 'test':
        vocab = {}
        s = 'xxxxxxxx${TEST_LINK}xxxxxxxxxxxxxx【30180退换货险】是由中国人寿提供给京东商城指定类目POP商户购买的，用于保障买家购买指定电器后发生厂保期内的性能故障时，提供30天内只退不换、180天只换不修服务的险种。【 白条 账单 - 消费 明细 】xxxxxx'
        s = '小金库转出至银行卡成功后又退回到小金库余额，可能是网络异常导致，请联系客服处理。${LXKF}'
        line = map_words(s, vocab)
        print("cut:"+ line)
        

