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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from six.moves import xrange
from os.path import join as pjoin
from PyLib import operate_file as opf
from levenshtein import align_all, align
from collections import OrderedDict
from multiprocessing import Pool
from process_lm import get_fst_for_group
import pywrapfst as fst
import sys



        
out_dir = ''
data_dir= ''
dev = ''
start = 0
end = 0
nthread=0
beam_size=100


def decode():
    # Prepare NLC data.
    global out_dir, data_dir, dev, start, end, nthread, beam_size
    folder_out = out_dir
    with open(pjoin(out_dir, dev + '.o.txt.' + str(start) + '_' + str(end)), 'r') as f_:
        lines = [ele[:-1].split('\t') for ele in f_.readlines()]
    with open(pjoin(data_dir, dev + '.z.txt'), 'r') as f_:
        truths = [ele.strip('\n') for ele in f_.readlines()]
    line_ids = np.loadtxt(pjoin(data_dir, dev + '.id'), dtype=int)
    if len(line_ids) != len(truths):
        raise 'ID number is not consistent with input file!'
    dict_id2line = OrderedDict()
    for i in range(line_ids.shape[0]):
        cur_id = line_ids[i]
        dict_id2line[cur_id] = i
    dict_id2group = opf.load_obj(pjoin(data_dir, dev + '.group'))
    max_size = 4
    process_group_id = dict_id2group[line_ids[start]]
    pro_group = []
    pro_truth = []
    pro_prob = []
    # pool = Pool(processes=nthread)
    # f_o = open(pjoin(folder_out, dev + '.e1.txt.' + str(start) + '_' + str(end)), 'w')
    for i in range(start, end):
        cur_out = [lines[k][0] for k in range(i * beam_size, (i + 1) * beam_size)]
        cur_prob = [float(lines[k][1]) for k in range(i * beam_size, (i + 1) * beam_size)]
        cur_truth = truths[i]
        cur_id = line_ids[i]
        cur_group_id = dict_id2group[cur_id]
        flag_process = 0
        if cur_group_id == process_group_id:
            pro_group.append(cur_out)
            pro_truth.append(cur_truth)
            pro_prob.append(cur_prob)
            if len(pro_group) == max_size:
                flag_process = 1
        else:
            flag_process = 1
        if flag_process:
            #ToDo: Add language Model
            #ToDo: fst.shortestdistance(fst.intersect(f2, lm), reverse=True)[0]
            #ToDo: fst.shorthestpath
            get_fst_for_group(i, pro_group, pro_prob, folder_out)
            # group_char_dis = 0
            # group_best_dis = 0
            # group_len = 0
            # list_fst = []
            # for j in range(len(pro_group)):
            #     cur_truth = pro_truth[j]
            #     outputs = pro_group[j]
            #     probs = pro_prob[j]
                # cur_truth_strip = cur_truth.strip()
                # len_y = len(cur_truth_strip)
                # cur_best = outputs[0]
                # best_dis = align(cur_truth_strip, cur_best)
                # best_char_dis, best_char_str = align_all(pool, cur_truth_strip,
                #                                          outputs, nthread,
                #                                          flag_char=1)
                # group_char_dis += best_char_dis
                # group_best_dis += best_dis
                # group_len += len_y
            # f_o.write('%d\t%d\t%d\n' % (group_char_dis, group_best_dis, group_len))
            pro_group = []
            pro_truth = []
            pro_prob = []
        if cur_group_id != process_group_id:
            pro_group.append(cur_out)
            pro_truth.append(cur_truth)
            process_group_id = cur_group_id
    # f_o.close()




def main():
    global out_dir, data_dir, dev, start, end, nthread, beam_size
    data_dir = '/scratch/dong.r/Dataset/OCR/' + sys.argv[1]
    out_dir = '/scratch/dong.r/Dataset/OCR/' + sys.argv[1] + '/' + sys.argv[2]
    dev = sys.argv[3]
    start = int(sys.argv[4])
    end = int(sys.argv[5])
    beam_size = int(sys.argv[6])
    decode()

main()
