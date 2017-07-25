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
from levenshtein import align, align_one2many
from collections import OrderedDict
from multiprocessing import Pool
from process_lm import *
import sys
from os.path import exists


out_dir = ''
data_dir= ''
lm_dir=''
dev = ''
start = 0
end = 0
nthread=100
beam_size=100
weight1=100
weight2=10


def decode():
    # Prepare NLC data.
    global out_dir, data_dir, lm_dir, dev, start, end, nthread, beam_size, weight1, weight2
    lm_name=lm_dir.split('/')[-1]
    new_out_dir = pjoin(out_dir, str(weight1) + '_' + str(weight2) + '_' + lm_name)
    if not exists(new_out_dir):
        os.makedirs(new_out_dir)
    with open(pjoin(out_dir, dev + '.o.txt.' + str(start) + '_' + str(end)), 'r') as f_:
        lines = [ele[:-1].split('\t') for ele in f_.readlines()]
    all_dis = np.loadtxt(pjoin(out_dir, dev + '.ec.txt.' + str(start) + '_' + str(end)), dtype=int)
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
    max_size = 100
    process_group_id = dict_id2group[line_ids[start]]
    pro_group = []
    pro_truth = []
    pro_prob = []
    pro_id=[]
    initialize(lm_dir)
    num_empty=0
    pool = Pool(processes=nthread, initializer=get_dict())
    f_o2 = open(pjoin(new_out_dir, dev + '.e2.txt.' + str(start) + '_' + str(end)), 'w')
    for i in range(end - start): 
        cur_pro_out = [lines[k][0].lower() for k in range(i * beam_size, (i + 1) * beam_size)]
        cur_pro_prob = []
        for k in range(i * beam_size, (i + 1) * beam_size):
            if len(lines[k]) == 1:
                cur_pro_prob.append(0.0)
            else:
                cur_pro_prob.append(float(lines[k][1]))
        cur_pro_truth = truths[i + start].lower()
        cur_id = line_ids[i]
        cur_group_id = dict_id2group[cur_id]
        flag_process = 0
        if cur_group_id == process_group_id:
            pro_group.append(cur_pro_out)
            pro_truth.append(cur_pro_truth)
            pro_prob.append(cur_pro_prob)
            pro_id.append(i)
            if len(pro_id) == max_size or i == end - start - 1:
                flag_process = 1
        else:
            flag_process = 1
        if flag_process and len(pro_id) > 0:
            #cur_str=''
            print('Get String ...')
            cur_str = get_fst_for_group_paral(pool, pro_group, pro_prob, pro_id, beam_size, start, new_out_dir, weight1, weight2)
            print(pro_id[0], pro_id[-1])
            cur_truth = ''.join(pro_truth).strip()
            print('Align String ...')
            dis = align(cur_str.strip(), cur_truth)
            f_o2.write('%d\t%d\t\n' % (dis, len(cur_truth)))
            pro_group = []
            pro_truth = []
            pro_id = []
            pro_prob = []
        if cur_group_id != process_group_id:
            pro_group.append( cur_pro_out)
            pro_truth.append(cur_pro_truth)
            pro_prob.append(cur_pro_prob)
            pro_id.append(i)
            process_group_id = cur_group_id
    f_o2.close()
    print(num_empty)



def main():
    global out_dir, data_dir, dev, start, end, nthread, beam_size, weight1, weight2, lm_dir
    data_dir = '/scratch/dong.r/Dataset/OCR/' + sys.argv[1]
    out_dir = '/scratch/dong.r/Dataset/OCR/' + sys.argv[1] + '/' + sys.argv[2]
    dev = sys.argv[3]
    start = int(sys.argv[4])
    end = int(sys.argv[5])
    beam_size = int(sys.argv[6])
    weight1=float(sys.argv[7])
    weight2=float(sys.argv[8])
    lm_dir=sys.argv[9]
    decode()

main()
