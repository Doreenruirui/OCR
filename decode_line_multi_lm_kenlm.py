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

import kenlm
import re
import os
import sys
import time
import numpy as np
from os.path import join as pjoin
from multiprocessing import Pool
from util_lm_kenlm import get_string_to_score
from levenshtein import align_pair, align


folder_data = ''
data_dir = ''
out_dir = ''
lm_dir = ''
dev = ''
lm_name = ''
start = 0
end = -1
lm = None


def initialize():
    global lm, folder_data, lm_dir
    lm = kenlm.LanguageModel(pjoin(folder_data, 'lm/char', lm_dir, 'train.arpa'))

def score_sent(paras):
    global lm
    thread_no, sent = paras
    sent = get_string_to_score(sent)
    return thread_no, lm.score(sent)

def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

def rank_sent(pool, sents):
    # res1 = score_sent([0, remove_nonascii(sents[0].replace('_', '-'))])
    # res2 = score_sent([0, remove_nonascii(sents[1].replace('_', '-'))])
    # new_sents = [remove_nonascii(ele) for ele in sents]
    #print(res1, res2)
    probs = np.ones(len(sents)) * -1
    results = pool.map(score_sent, zip(np.arange(len(sents)), sents))
    max_str = ''
    max_prob = float('-inf')
    for tid, score in results:
        cur_prob = score
        #cur_prob = np.power(10, -score)
        probs[tid] = cur_prob
        if cur_prob > max_prob:
            max_prob = cur_prob
            max_str = sents[tid]
    return max_str, max_prob, probs


def decode():
    global folder_data, data_dir, out_dir, lm_dir, dev, start, end, lm_name, lm
    data_dir = pjoin(folder_data, data_dir)
    folder_out = pjoin(data_dir, out_dir)
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    tic = time.time()
    with open(pjoin(data_dir, dev + '.x.txt'), 'r') as f_:
        lines = [ele for ele in f_.readlines()]
    with open(pjoin(data_dir, dev + '.y.txt'), 'r') as f_:
        truths = [ele.strip().lower() for ele in f_.readlines()]
    f_o = open(pjoin(folder_out, dev + '.' + str(lm_name) + '.kenlm.' + 'ec.txt.' + str(start) + '_' + str(end)), 'w')
    f_b = open(pjoin(folder_out, dev + '.' + str(lm_name) + '.kenlm.' + 'o.txt.' + str(start) + '_' + str(end)), 'w')
    pool = Pool(100, initializer=initialize())
    for line_id in range(start, end):
        line = lines[line_id]
        cur_truth = truths[line_id]
        sents = [ele for ele in line.strip('\n').split('\t')][:100]
        sents = [ele.strip() for ele in sents if len(ele.strip()) > 0]
        if len(sents) > 0:
            if 'low' in lm_dir:
                sents = [ele.lower() for ele in sents]
            best_sent, best_prob, probs = rank_sent(pool, sents)
            best_dis = align(cur_truth, best_sent.lower())
            f_o.write(str(best_dis) + '\t' + str(len(cur_truth)) + '\n')
            f_b.write(best_sent + '\n')
        else:
            f_o.write(str(len(cur_truth)) + '\t' + str(len(cur_truth)) + '\n')
            f_b.write('' + '\n')
        if line_id % 100 == 0:
            toc = time.time()
            print(toc - tic)
            tic = time.time()
    f_o.close()
    f_b.close()


def main():
    global folder_data, data_dir, out_dir, lm_dir, dev, start, end, lm_name
    folder_data = sys.argv[1]
    data_dir = sys.argv[2]
    out_dir = sys.argv[3]
    lm_dir = sys.argv[4]
    lm_name = sys.argv[5]
    dev = sys.argv[6]
    start = int(sys.argv[7])
    end = int(sys.argv[8])
    decode()


if __name__ == "__main__":
    main()
