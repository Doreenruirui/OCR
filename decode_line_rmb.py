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
from levenshtein import align


folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR/'
data_dir = ''
out_dir = ''
lm_dir = ''
dev = ''
lm_name = ''
start = 0
end = -1
w1 = 0.0
w2 = 0.0

def mbr(paras):
    thread_no, sents, probs = paras
    min_id = -1
    min_v = -1
    for cur_id in range(len(sents)):
        cur_sent = sents[cur_id]
        sum_v = 0
        for i in range(len(sents)):
            if i != cur_id:
                sent = sents[i]
                dis = align(cur_sent, sent)
                sum_v += np.exp(probs[i]) * dis
        if min_v < sum_v:
            min_v = sum_v
            min_id = cur_id
    return thread_no, min_id, min_v


def evaluate():
    global folder_data, data_dir, out_dir, lm_dir, dev, start, end, w1, w2
    data_dir = pjoin(folder_data, data_dir)
    folder_out = pjoin(data_dir, out_dir)
    lines = []
    for line in file(pjoin(data_dir, dev + '.x.txt')):
        lines.append([ele.strip() for ele in line.strip('\n').split('\t')][0])
    truths = []
    for line in file(pjoin(data_dir, dev + '.y.txt')):
        truths.append(line.strip('\n'))
    outputs = []
    for line in file(pjoin(folder_out, dev + '.o.txt.' + str(start) + '_' + str(end))):
        outputs.append(line.strip())
    output_probs = []
    for line in file(pjoin(folder_out, dev + '.p.txt.' + str(start) + '_' + str(end))):
        output_probs.append(float(line.strip()))
    f_ = open(pjoin(folder_out, dev +  '.' + 'mbr.ec.txt.' + str(start) + '_' + str(end)), 'w')
    pool = Pool(100)
    nsent = 1
    pro_paras = []
    pro_truth = []

    for line_id in range(start, end):
        cur_start = (line_id - start) * 100
        cur_end = (line_id + 1 - start) * 100
        # cur_line = lines[line_id]
        # cur_truth = truths[line_id]
        pro_paras.append([line_id % 100, outputs[cur_start: cur_end], output_probs[cur_start: cur_end]])
        pro_truth.append(truths[line_id])
        if (line_id + 1) % nsent == 0:
            res = pool.map(mbr, pro_paras)
            new_res = [0 for _ in range(nsent)]
            pro_paras = []
            pro_truth = []
            for thread_no, min_id, min_v in res:
                new_res[thread_no] = align(pro_paras[1][thread_no][min_id], pro_truth[thread_no])
            for ele in new_res:
                f_.write(str(ele) + '\n')
    f_.close()


def main():
    global data_dir, out_dir, lm_dir, dev, start, end, w1, w2
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    # lm_name = sys.argv[4]
    dev = sys.argv[3]
    start = int(sys.argv[4])
    end = int(sys.argv[5])
    evaluate()


if __name__ == "__main__":
    main()
