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
from util_lm import get_fst_for_group_paral, initialize_score
from levenshtein import align_pair, align


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

def get_line2group():
    global folder_data, data_dir, out_dir, lm_dir, dev, start, end, w1, w2
    data_dir = pjoin(folder_data, data_dir)
    file_info = pjoin(data_dir, dev + '.info.txt')
    group_no = 0
    last_end = 0
    line_id = 0
    cur_key = ''
    line2group = []
    for line in file(file_info):
        date, seq, start, end = line.strip('\n').split('\t')
        start = int(start)
        end = int(end)
        if line_id == 0:
            cur_key = date + '-' + seq
        new_key = date + '-' + seq
        if new_key != cur_key:
            group_no += 1
            cur_key = new_key
        elif start != last_end + 1:
            group_no += 1
        last_end = end
        line2group.append(group_no)
        line_id += 1
    with open(pjoin(data_dir, dev + '.line2group'), 'w') as f_:
        for line in line2group:
            f_.write(str(line) + '\n') 

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
        if 'low' in lm_dir:
            outputs.append(line.strip().lower())
        else:
            outputs.append(line.strip())
    output_probs = []
    for line in file(pjoin(folder_test, dev + '.p.txt.' + str(start) + '_' + str(end))):
        output_probs.append(float(line.strip()))
    line2group = []
    for line in file(pjoin(data_dir, dev + '.line2group')):
        line2group.append(int(line.strip('\n')))
    f_ = open(pjoin(folder_out, dev +  '.' + 'ec.txt.' + str(start) + '_' + str(end)), 'w')
    pool = Pool(100, initializer=initialize_score(pjoin(folder_data, 'voc'), pjoin(folder_data, 'lm/char', lm_dir)))
    pro_group_id = line2group[start]
    pro_group = []
    pro_prob = []
    pro_truth = []
    pro_lines = []
    initialize_score(pjoin(folder_data, 'voc'), pjoin(folder_data, 'lm/char', lm_dir))
    for line_id in range(start, end):
        cur_start = (line_id - start) * 100
        cur_end = (line_id + 1 - start) * 100
        cur_group_id = line2group[line_id]
        cur_group = outputs[cur_start: cur_end]
        cur_prob = output_probs[cur_start: cur_end]
        cur_line = lines[line_id]
        cur_truth = truths[line_id]
        if cur_group_id != pro_group_id:
            best_str = get_fst_for_group_paral(pool, pro_group, pro_prob, pro_group_id, folder_tmp, w1, w2)
            len_truth = len(''.join(pro_truth))
            strip_truth = ''.join([ele.strip() for ele in pro_truth])
            cur_dec = ''.join([ele[0].strip() for ele in pro_group]).lower()
            dis1 = align(best_str.lower(), ''.join(pro_truth).lower())
            dis2 = align(''.join(pro_lines).lower(), strip_truth.lower())
            dis3 = align(cur_dec, strip_truth.lower())
            pro_group = []
            pro_truth = []
            pro_prob = []
            pro_lines = []
            pro_group_id = cur_group_id
            f_.write(str(dis1) + '\t' + str(len_truth) + '\t' + str(dis2) +  '\t' + str(dis3) + '\t' + str(len(strip_truth)) + '\n')
        pro_group.append(cur_group)
        pro_prob.append(cur_prob)
        pro_truth.append(cur_truth)
        pro_lines.append(cur_line)
    f_.close()


def main():
    global data_dir, out_dir, lm_dir, dev, start, end, w1, w2
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    lm_dir = sys.argv[3]
    # lm_name = sys.argv[4]
    dev = sys.argv[4]
    start = int(sys.argv[5])
    end = int(sys.argv[6])
    w1 = float(sys.argv[7])
    w2 = float(sys.argv[8])
    #get_line2group()
    evaluate()


if __name__ == "__main__":
    main()
