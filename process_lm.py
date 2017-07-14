##TODO: Prepare the Lexicon Transducer

import pywrapfst as fst
import numpy as np
import subprocess
# from util_thread import split_task


dict_char2id = {}
dict_id2char = {}
dict_word2id = {}
dict_id2word = {}
file_voc = '/scratch/dong.r/Dataset/OCR/voc/ascii.syms'
tbl = fst.SymbolTable.read_text(file_voc)
lm = fst.Fst.read('/scratch/dong.r/Dataset/OCR/lm/char/richmond/train.mod')
concat_f=fst.Fst.read('/scratch/dong.r/Dataset/OCR/lm/concat.fst')
list_res = []


def split_task(pool, task, list_paras):
    list_input = []
    ntask = len(list_paras[0])
    ninput = len(list_paras)
    for x in range(ntask):
        cur_input = (x, )
        for i in range(ninput):
            cur_input += (list_paras[i][x], )
        list_input.append(cur_input)
    results = pool.map(task, list_input)
    res = [None for i in range(ntask)]
    for i in range(ntask):
        thread_num, cur_res = results[i]
        res[thread_num] = cur_res
    if None in res:
        raise 'Something Wrong!'
    return res


def get_dict():
    global dict_id2char, dict_char2id, lm
    for line in file(file_voc):
        char, cid = line.strip().split('\t')
        dict_char2id[char] = int(cid)
        dict_id2char[int(cid)] = char

def get_voc():
    global dict_word2id, dict_id2word
    for line in file('./lm/word.syms'):
        word, wid = line.strip().split('\t')
        dict_char2id[word] = int(wid)
        dict_id2char[int(wid)] = word


def sentence_fst(sent, pb):
    f = fst.Fst()
    s = f.add_state()
    f.set_start(s)
    if len(sent) == 0:
        n = f.add_state()
        f.set_final(n, 0.0)
        cur_id = dict_char2id['<epsilon>']
        f.add_arc(s, fst.Arc(cur_id, cur_id, -pb, n))
        return f
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
    set_symbol(f)
    # f.write()
    return f



# def thread_sentence_fst(paras):
#     thread_no, sent, pb = paras
#     f = sentence_fst(sent, pb)
#     #list_res[thread_no] = f
#     print thread_no


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


def list_sentence_fst(outputs, probs, w):
    global dict_char2id, dict_id2char
    if len(dict_id2char) == 0 or len(dict_char2id) == 0:
        get_dict()
    list_fst = []
    for i in range(len(outputs)):
        sent = outputs[i]
        pb = probs[i] * w
        list_fst.append(sentence_fst(sent, pb))
    final = combine_fst(list_fst)
    return final


def get_fst_for_output(outputs, probs, w):
    global dict_char2id, dict_id2char
    if len(dict_id2char) == 0 or len(dict_char2id) == 0:
        get_dict()
    list_fst = []
    for i in range(len(outputs)):
        sent = outputs[i]
        pb = probs[i] * w
        list_fst.append(sentence_fst(sent, pb))
    final = combine_fst(list_fst)
    return final


def concat_fst(list_fst):
    final = list_fst[0]
    # concat_symbol = ['-', '<space>', '<epsilon>']
    for f in list_fst[1:]:
        final = final.concat(concat_f)
        final = final.concat(f)
    # for f in list_fst[1:]:
    #     cur_nstate = final.num_states()
    #     final = final.concat(f)
    #     final.delete_arcs(cur_nstate - 1)
    #     for sym in concat_symbol:
    #         cur_id = dict_char2id[sym]
    #         final.add_arc(cur_nstate - 1, fst.Arc(cur_id, cur_id, 0.0, cur_nstate))
    return final


def get_fst_for_group_sent(group, group_prob, w):
    #get_fst_for_group 1 4 richmond/0/0/50/train/100 test.o.txt.0_1000 1000
    list_fst = []
    for i in range(len(group)):
        cur_group = group[i]
        cur_prob = group_prob[i]
        cur_prob = [ele * w for ele in cur_prob]
        list_fst.append(list_sentence_fst(cur_group, cur_prob, w))
    final = concat_fst(list_fst)
    set_symbol(final)
    final=fst.shortestpath(fst.intersect(final, lm)).rmepsilon()
    string= print_path(final)
    return string


def get_fst_for_group(pro_id, file_no, folder_data, file_name, w):
    #get_fst_for_group 1 4 richmond/0/0/50/train/100 test.o.txt.0_1000 1000
    start_line=pro_id[0]
    subprocess.call(['get_fst_for_group', str(pro_id[0] + 1), str(pro_id[-1] + 1), folder_data, file_name])
    subprocess.call(['sh', '/home/dong.r/nlc-master/run_group.%d.%d' % (file_no, start_line)])
    subprocess.call(['rm', '/home/dong.r/nlc-master/run_group.%d.%d' % (file_no, start_line)])
    f = read_fst(folder_data + '/fst.tmp.0.%d.score' % start_line)
    return print_path(f)

    # list_fst = []
    # for i in range(len(group)):
    #     cur_group = group[i]
    #     cur_prob = group_prob[i]
    #     cur_prob = [ele * w for ele in cur_prob]
    #     list_fst.append(list_sentence_fst(cur_group, cur_prob, w))
    # final = concat_fst(list_fst)
    # set_symbol(final)
    # final=fst.shortestpath(fst.intersect(final, lm)).rmepsilon()
    # string1 = print_path(final)
    # fst2 = read_fst('/scratch/dong.r/Dataset/OCR/richmond/0/0/50/train/100/fst.tmp.0.%d.score' % pro_id)
    # string2 = print_path(fst2)




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
    f2 = sentence_fst('I am a good gir', np.log(0.2))
    f3 = sentence_fst('l! Rig', np.log(0.3))
    f4 = sentence_fst('ht?', np.log(0.1))
    f5 = concat_fst([f2, f3, f4])
    # f3 = combine_fst([f1, f2])
    # f4 = normalize_weight(f3)
    print f5

# test()
