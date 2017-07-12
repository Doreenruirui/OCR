##TODO: Prepare the Lexicon Transducer

import pywrapfst as fst
import numpy as np
from os.path import join as pjoin
from multiprocessing import Pool


dict_char2id = {}
dict_id2char = {}
dict_word2id = {}
dict_id2word = {}
tbl = fst.SymbolTable.read_text('/scratch/dong.r/Dataset/OCR/voc/ascii.syms')


def get_dict():
    global dict_id2char, dict_char2id
    for line in file('/scratch/dong.r/Dataset/OCR/voc/ascii.syms'):
        char, cid = line.strip().split('\t')
        dict_char2id[char] = int(cid)
        dict_id2char[int(cid)] = char

def get_voc():
    global dict_word2id, dict_id2word
    for line in file('./lm/word.syms'):
        word, wid = line.strip().split('\t')
        dict_char2id[word] = int(wid)
        dict_id2char[int(wid)] = word


def get_fst_from_sentence(sent, pb):
    global dict_id2char, dict_char2id
    f = fst.Fst()
    s = f.add_state()
    f.set_start(s)
    for char in sent:
        n = f.add_state()
        if char != ' ':
            cur_id = dict_char2id[char]
        else:
            cur_id = dict_char2id['<space>']
        if s == 0:
            f.add_arc(s, fst.Arc(cur_id, cur_id, -pb, n))
        else:
            f.add_arc(s, fst.Arc(cur_id, cur_id, 0.0, n))
        s = n
    f.set_final(n, 0.0)
    return f


def combine_fst(list_fst):
    final = list_fst[0]
    for cur_fst in list_fst[1:]:
        final = fst.determinize(cur_fst.union(final).rmepsilon())
    return final.minimize()


def normalize_weight(f):
    for state in f.states():
        cur_sum = 0
        cur_arcs = []
        for arc in f.arcs(state):
            next = arc.nextstate
            weight = float(arc.weight.to_string())
            ilabel = arc.ilabel
            olabel = arc.olabel
            cur_sum += weight
            cur_arcs.append((next, weight, ilabel, olabel))
        f.delete_arcs(state)
        for arc in cur_arcs:
            f.add_arc(state, fst.Arc(arc[2], arc[3], arc[1] * 1. / cur_sum, arc[0]))
    return f


def log_weight(f):
    for state in f.states():
        cur_arcs = []
        for arc in f.arcs(state):
            nexts = arc.nextstate
            weight = np.log(float(arc.weight.to_string()))
            ilabel = arc.ilabel
            olabel = arc.olabel
            cur_arcs.append((nexts, weight, ilabel, olabel))
        f.delete_arcs(state)
        for arc in cur_arcs:
            f.add_arc(state, fst.Arc(arc[2], arc[3], arc[1], arc[0]))
    return f


def get_fst_for_output(outputs, probs, w):
    global dict_char2id, dict_id2char
    if len(dict_id2char) == 0 or len(dict_char2id) == 0:
        get_dict()
    list_fst = []
    for i in range(len(outputs)):
        sent = outputs[i]
        pb = probs[i] * w
        list_fst.append(get_fst_from_sentence(sent, pb))
    final = combine_fst(list_fst)
    return final


def concat_fst(list_fst):
    final = list_fst[0]
    concat_symbol = ['-', '<space>', '<epsilon>']
    for f in list_fst[1:]:
        cur_nstate = final.num_states()
        final = final.concat(f)
        final.delete_arcs(cur_nstate - 1)
        for sym in concat_symbol:
            cur_id = dict_char2id[sym]
            final.add_arc(cur_nstate - 1, fst.Arc(cur_id, cur_id, 0.0, cur_nstate))
    return final

def get_fst_for_group(cur_group_id, group, group_prob, folder_out):
    list_fst = []
    for i in range(len(group)):
        cur_group = group[i]
        cur_prob = group_prob[i]
        list_fst.append(get_fst_for_output(cur_group, cur_prob))
    final = concat_fst(list_fst)
    #tbl = fst.SymbolTable.read_text('/scratch/dong.r/Dataset/OCR/voc/ascii.syms')
    final.set_input_symbols(tbl)
    final.set_output_symbols(tbl)
    final.write(pjoin(folder_out, 'f' + str(cur_group_id) + '.fst'))
    # subprocess.call()
    # subprocess.call(["dot", "-Tsvg", pjoin(folder_out, 'f' + str(cur_group_id) + '.gv'), ">", pjoin(folder_out, 'f' + str(cur_group_id) + '.svg')])

    # final.draw(pjoin(folder_out, 'f' + str(cur_group_id) + '.gv'), portrait=True)
    # subprocess.call(["dot", "-Tsvg", pjoin(folder_out, 'f' + str(cur_group_id) + '.gv'), ">", pjoin(folder_out, 'f' + str(cur_group_id) + '.svg')])


def get_fst_for_word(word):
    global dict_id2char, dict_char2id
    f = fst.Fst()
    s = f.add_state()
    f.set_start(s)
    n = f.add_state()
    f.add_arc(s, fst.Arc(dict_char2id[word[0]], dict_word2id[word], 1, n))
    s = n
    for char in word[1:]:
        n = f.add_state()
        if char != ' ':
            cur_id = dict_char2id[char]
        else:
            cur_id = dict_char2id['<space>']
        f.add_arc(s, fst.Arc(cur_id, dict_word2id['<epsilon>'], 1, n))
        s = n
    f.set_final(n, 1)
    return f


def read_fst_from_text(filename):
    dict_state = {}
    state_id = 0
    f = fst.Fst()
    for line in file(filename):
        line = line.strip().split('\t')
        s1 = int(line[0])
        if len(line) == 5:
            s2 = int(line[1])
        if s1 not in dict_state:
            dict_state[s1] = state_id
            state_id += 1
            f.add_state()
        if s2 not in dict_state:
            dict_state[s2] = state_id
            state_id += 1
            f.add_state()
    f.set_start(0)
    for line in file(filename):
        line = line.strip().split('\t')
        s1 = int(line[0])
        sid1 = dict_state[s1]
        s = f.add_state()
        if len(line) == 1:
            f.set_final(sid1, 0)
        elif len(line) == 2:
            f.set_final(sid1, float(line[1]))
        else:
            s2 = int(line[1])
            sid2 = dict_state[s2]
            il = line[2]
            ol = line[3]
            w = float(line[4])
            f.add_arc(sid1, fst.Arc(il, ol, w, sid2))
    return f


def print_path(f):
    get_dict()
    cur_str = ''
    for state in f.states():
        for arc in f.arcs(state):
            cur_str += dict_id2char[arc.ilabel]
    cur_str = cur_str.replace('<space>', ' ')
    cur_str = cur_str[::-1]
    return cur_str


def read_fst(filename):
    v = fst.Fst.read(filename)
    return v


def set_symbol(f):
    f.set_input_symbols(tbl)
    f.set_output_symbols(tbl)
    return f


# def build_lexicon_fst(vocfile):
#     list_fst = []
#     for line in vocfile:
#         word = line.strip()
#         list_fst.append(get_fst_for_word(word))
#     f = combine_fst(list_fst)
#     final_state = f.num_states() - 1
#     cur_id = 0
#     for state in f.states():
#         f.add_arc(state, fst.Arc(dict_char2id['<phi>'], dict_char2id['<phi>'], 1, final_state))
#         cur_id += 1



def test():
    get_dict()
    f2 = get_fst_from_sentence('I am a good gir', np.log(0.2))
    f3 = get_fst_from_sentence('l! Rig', np.log(0.3))
    f4 = get_fst_from_sentence('ht?', np.log(0.1))
    f5 = concat_fst([f2, f3, f4])
    # f3 = combine_fst([f1, f2])
    # f4 = normalize_weight(f3)
    print f5

# test()
