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

VOCAB_FILE = "vocab.json"

class StringPiece(object):
    def __init__(self, string, is_orig=True):
        self.string = string
        self.is_orig = is_orig
    def __repr__(self):
        return "[string={}, is_orig={}]".format(self.string, self.is_orig)

def map_iter(input_line, vocab):
    line = input_line.strip()
    piece_list = [StringPiece(line)]
    new_list = []
    for p in PATTERN_LIST:
        for piece in piece_list:
            #print("map_iter piece:{} for p:{}".format(piece, p)) 
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


def map_words_recursive(input_line, vocab, depth=0):
    if depth >= len(PATTERN_LIST):
        return [input_line]
    p = PATTERN_LIST[depth]
    result = p.search(p)
    if not result:
        return [input_line]
    left = input_line[:result.start]
    right = input_line[result.end:]
    depth += 1
    
    left_result = map_words_recursive(left, vocab, depth)
    right_result = map_words_recursive(right, vocab, depth)
    return left_result + [symbol] + right_result

def map_words_raw(input_line, vocab, depth=0):
    line = input_line.strip()
    cnt = 0; print("line:" + line + " depth:" + str(depth))
    match_flag = False; final_string = line
    for p in PATTERN_LIST:
        result = p.search(line)
        print("0 {} match:{} for p:{}".format(line, result, p))
        last_end = 0
        total = len(line)
        end = 0
        piece_list = []
        cnt += 1
        while result:
            match_flag = True; cnt += 1
            start = result.start(); end = result.end()
            print("line:{} match:{} s:{} e:{}".format(line[last_end:], result.group(0), start, end))
            if start > last_end:
                piece_list.append(StringPiece(line[last_end:start])); print("1:piece_list:" + str(piece_list))
            w = result.group(0)
            if w not in vocab:
                symbol = SYMBOL_FMT.format(len(vocab))
                vocab[w] = symbol; print("add new symbol:" + symbol)
            else:
                symbol = vocab[w]
            piece_list.append(StringPiece(" {} ".format(symbol), False)); print("2:piece_list:" + str(piece_list))
            last_end = end
            if end < total:
                result = p.search(line, end)
            if cnt > 10:
                print("reach maximum")
                return 
        if end < total and end > 0:
            piece_list.append(StringPiece(line[end:total])); print("3:piece_list:" + str(piece_list))
        print("piece_list:" + str(piece_list))
        final_string = ""
        for piece in piece_list:
            if not piece.is_orig:
                final_string += piece.string
            else:
                final_string += map_words_raw(piece.string, vocab, depth+1)
        if match_flag: break
    print("final_string:" + final_string)
    return final_string    
    #words = jieba.lcut(line)
    #return ' '.join(words)


def map_words(input_line, vocab):
    line = input_line.strip()
    for p in PATTERN_LIST:
        result = p.findall(line)
        local_vocab = {}
        result = list(set(result))
        for i,r in enumerate(result):
            if r not in vocab:
                symbol = SYMBOL_FMT.format(len(vocab))
                vocab[r] = symbol
                local_vocab[r] = symbol
            else:
                local_vocab[r] = vocab[r]

        for k,v in local_vocab.items():
            line = line.replace(k, " " + v + " ")
        
    words = jieba.lcut(line)
    return ' '.join(words)


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
                l = map_iter(line, vocab)
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

        

