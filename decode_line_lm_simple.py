# Copyright 2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import re
import os
import sys
import time
import numpy as np
from os.path import join as pjoin
from multiprocessing import Pool
from util_lm_kenlm import get_string_to_score
import kenlm
from levenshtein import align_pair, align


folder_data = '/scratch/dong.r/Dataset/OCR/'
data_dir = ''
out_dir = ''
lm_dir1 = ''
lm_dir2 = ''
dev = ''
lm_name = ''
start = 0
end = -1
w1 = 0.0
w2 = 0.0


def initialize():
    global lm1, lm2, folder_data, lm_dir1, lm_dir2
    lm1 = kenlm.LanguageModel(pjoin(folder_data, 'lm/word', lm_dir1, 'train.binary'))
    lm2 = kenlm.LanguageModel(pjoin(folder_data, 'lm/char', lm_dir2, 'train.binary'))


def score_sent_word(paras):
    global lm1
    thread_no, sent = paras
    return thread_no, lm1.score(' ' + ' '.join(sent.split(' ')[1:-1]) + ' ')


def score_sent_char(paras):
    global lm2
    thread_no, sent = paras
    sent = get_string_to_score(sent)
    return thread_no, lm2.score(sent)


def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def rank_sent_word(pool, sents):
    probs = np.ones(len(sents)) * -1
    results = pool.map(score_sent_word, zip(np.arange(len(sents)), sents))
    for tid, score1 in results:
        probs[tid] = score1
    return probs

def rank_sent_char(pool, sents):
    probs = np.ones(len(sents)) * -1
    results = pool.map(score_sent_char, zip(np.arange(len(sents)), sents))
    for tid, score1 in results:
        probs[tid] = score1
    return probs


def evaluate():
    global folder_data, data_dir, out_dir, lm_dir, dev, start, end, w1, w2
    data_dir = pjoin(folder_data, data_dir)
    folder_test = pjoin(data_dir, out_dir)
    folder_out = pjoin(folder_test, str(w1) + '_' + str(w2))
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    folder_tmp = pjoin(folder_out, 'tmp')
    if not os.path.exists(folder_tmp):
        os.makedirs(folder_tmp)
    lines = []
    for line in file(pjoin(data_dir, dev + '.x.txt')):
        lines.append([ele.strip() for ele in line.strip('\n').split('\t')][0])
    truths = []
    for line in file(pjoin(data_dir, dev + '.y.txt')):
        truths.append(line.strip('\n'))
    outputs = []
    for line in file(pjoin(folder_test, dev + '.o.txt.' + str(start) + '_' + str(end))):
        outputs.append(line.strip())
    output_probs = []
    for line in file(pjoin(folder_test, dev + '.p.txt.' + str(start) + '_' + str(end))):
        output_probs.append(float(line.strip()))
    line2group = []
    for line in file(pjoin(data_dir, dev + '.line2group')):
        line2group.append(int(line.strip('\n')))
    f_ = open(pjoin(folder_out, dev +  '.' + 'ec.txt.' + str(start) + '_' + str(end)), 'w')
    pool = Pool(100, initializer=initialize())
    initialize()
    for line_id in range(start, end):
        cur_start = (line_id - start) * 100
        cur_end = (line_id + 1 - start) * 100
        cur_group = outputs[cur_start: cur_end]
        cur_prob = output_probs[cur_start: cur_end]
        cur_line = lines[line_id]
        cur_truth = truths[line_id]
        probs_word = rank_sent_word(pool, cur_group)
        #dict_probs = {}
        #for ele in probs_word:
        #    dict_probs[ele] = dict_probs.get(ele, 0) + 1
        probs_sent = rank_sent_char(pool, cur_group)
        #probs_word = rank_sent_word(pool, cur_group) / [len(ele.split(' ')[1:-1]) for ele in cur_group]
        #probs_sent = rank_sent_char(pool, cur_group) / [len(ele) for ele in cur_group]
        ocr_dis = align(cur_line, cur_truth)
        top_dis = align(cur_group[0], cur_truth)
        new_prob = cur_prob + w1 * probs_word + w2 * probs_sent
        max_prob = float('-inf')
        max_str = ''
        for i in range(len(cur_group)):
            if max_prob < new_prob[i]:
                max_prob = new_prob[i]
                max_str = cur_group[i]
        best_dis = align(max_str, cur_truth)
        print ocr_dis, top_dis, best_dis
    f_.close()


def main():
    global data_dir, out_dir, lm_dir1, lm_dir2, dev, start, end, w1, w2
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    lm_dir1 = sys.argv[3]
    lm_dir2 = sys.argv[4]
    # lm_name = sys.argv[4]
    dev = sys.argv[5]
    start = int(sys.argv[6])
    end = int(sys.argv[7])
    w1 = float(sys.argv[8])
    w2 = float(sys.argv[9])
    evaluate()


if __name__ == "__main__":
    main()
