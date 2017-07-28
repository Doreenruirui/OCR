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



import os
import sys
import time
import numpy as np
from os.path import join as pjoin
from multiprocessing import Pool
from process_lm import score_sent, initialize_score
from levenshtein import align_pair, align


folder_data = ''
data_dir = ''
out_dir = ''
lm_dir = ''
dev = ''
start = 0
end = -1


def rank_sent(pool, sents):
    sents = [ele.replace('_', '-') for ele in sents]
    res1 = score_sent([sents[0]])
    #res2 = score_sent([sents[1].replace('-', '_')])
    #print(res1, res2)
    probs = np.ones(len(sents)) * -1
    results = pool.map(score_sent, zip(np.arange(len(sents)), sents))
    max_str = ''
    max_prob = -1
    for tid, score in results:
        cur_prob = np.power(10, -score)
        probs[tid] = cur_prob
        if cur_prob > max_prob:
            max_prob = cur_prob
            max_str = sents[tid]
    return max_str, max_prob, probs


def decode():
    global folder_data, data_dir, out_dir, lm_dir, dev, start, end
    data_dir = pjoin(folder_data, data_dir)
    folder_out = pjoin(data_dir, out_dir)
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    tic = time.time()
    with open(pjoin(data_dir, dev + '.x.txt'), 'r') as f_:
        lines = [ele.strip() for ele in f_.readlines()]
    with open(pjoin(data_dir, dev + '.y.txt'), 'r') as f_:
        truths = [ele.strip() for ele in f_.readlines()]
    f_o = open(pjoin(folder_out, dev + '.ec4.txt.' + str(lm_dir) + '.' + str(start) + '_' + str(end)), 'w')
    pool = Pool(10, initializer=initialize_score(pjoin(folder_data, 'voc'), pjoin(folder_data, 'lm/char', lm_dir)))
    initialize_score(pjoin(folder_data, 'voc'), pjoin(folder_data, 'lm/char', lm_dir))
    for line_id in range(start, end):
        line = lines[line_id]
        cur_truth = truths[line_id]
        sents = [ele for ele in line.strip('\n').split('\t')]
        #sents = [ele for ele in line.strip('\n').split('\t')][0:100]
        if len(sents) > 0:
            best_sent, best_prob, probs = rank_sent(pool, sents)
            best_dis = align(cur_truth, best_sent)
            cur_dis = align_pair(pool, [cur_truth for _ in sents], sents)
            f_o.write(str(best_dis) + '\t' + '\t'.join([str(dis) + '\t' + str(prob) for dis, prob in zip(cur_dis, probs)]))
        else:
            f_o.write(str(len(cur_truth)) + '\t' + '\t'.join([str(len(cur_truth)) + '\t1' for _ in sents]))
        if line_id % 100 == 0:
            toc = time.time()
            print(toc - tic)
            tic = time.time()
    f_o.close()


def main():
    global folder_data, data_dir, out_dir, lm_dir, dev, start, end
    folder_data = sys.argv[1]
    data_dir = sys.argv[2]
    out_dir = sys.argv[3]
    lm_dir = sys.argv[4]
    dev = sys.argv[5]
    start = int(sys.argv[6])
    end = int(sys.argv[7])
    decode()


if __name__ == "__main__":
    main()
