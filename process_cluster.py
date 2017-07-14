from PyLib import operate_file as opf
from levenshtein import align_pair
from my_utils import error_rate_line
from collections import OrderedDict
from os.path import join, exists
import  numpy as np
import json
import os
import re
import sys


folder_data = '/scratch/dong.r/Dataset/OCR/eng'
dict_cat2name = {'m': 'monograph', 'p': 'periodical', 'o': 'other'}


def load_pairs_folder():
    keys = ['b1', 'e1', 'b2', 'e2']
    pair_file = {}
    folder_name = {}
    for cat in dict_cat2name:
        folder_name[cat] = join(folder_data, dict_cat2name[cat])
        if not exists(folder_name[cat]):
            os.makedirs(folder_name[cat])
        pair_file[cat] = {}
        for key in ['x', 'y', 'info']:
            pair_file[cat][key] = open(join(folder_name[cat], 'pair_' + key), 'w')
    files = [f for f in os.listdir(join(folder_data, 'pass.json'))]
    pair_id = 0
    line_id = 0
    num_other = 0
    for fname in files:
        if fname.startswith('part-'):
            for line in file(join(folder_data, 'pass.json', fname)):
                cur_dict = json.loads(line.strip())
                for pair in cur_dict['pairs']:
                    str1 = re.sub('\n', ' ', pair['_1'].strip())
                    str2 = re.sub('\n', ' ', pair['_2'].strip())
                    if 'periodical' not in cur_dict['id2'] and 'monograph' not in cur_dict['id2']:
                        cur_key = 'o'
                        num_other += 1
                        cur_info = cur_dict['id1'] + '\t' \
                               +  cur_dict['id2'] + '\t0\t' \
                               + '\t'.join(map(str, [cur_dict[ele] for ele in keys]))
                    else:
                        list_info = cur_dict['id2'].split('_')
                        if list_info[1] == 'periodical':
                            cur_key = 'p'
                        elif list_info[1] == 'monograph':
                            cur_key = 'm'
                        if len(list_info) == 3:
                            cur_task = 0
                        else:
                            if list_info[-2] == 'task1':
                                cur_task = 1
                            elif list_info[-2] == 'task2':
                                cur_task = 2
                        cur_info = cur_dict['id1'] + '\t' \
                               +  list_info[-1] + '\t'\
                               + str(cur_task) + '\t' \
                               + '\t'.join(map(str, [cur_dict[ele] for ele in keys]))
                    pair_file[cur_key]['x'].write(str1.encode('utf-8') + '\n')
                    pair_file[cur_key]['y'].write(str2.encode('utf-8') + '\n')
                    pair_file[cur_key]['info'].write(cur_info + '\n')
                    pair_id += 1
                line_id += 1
    print num_other
    np.savetxt(join(folder_data, 'statistics'), np.asarray([line_id, pair_id]))
    for cat in pair_file:
        for fh in pair_file[cat]:
            fh.close()


def compute_error_rate(cat_name, begin, end):
    folder_name = join(folder_data, dict_cat2name[cat_name])
    file_ocr = join(folder_name, 'pair_x')
    file_truth = join(folder_name, 'pair_y')
    folder_error = join(folder_name, 'error')
    if not exists(folder_error):
        os.makedirs(folder_error)
    file_error = join(folder_error, 'error_%d_%d' % (begin, end))
    error_rate_line(file_error, file_ocr, file_truth, begin, end)


def get_train_data(cat_name):
    folder_cat = dict_cat2name[cat_name]
    dict_page_pair = OrderedDict()
    line_id = 0
    for line in file(join(folder_data, folder_cat, 'pair_info')):
        list_items = line.strip().split('\t')
        key = list_items[1]
        if key not in dict_page_pair:
            dict_page_pair[key] = []
        dict_page_pair[key].append(line_id)
        line_id += 1
    with open(join(folder_data, folder_cat, 'pair_index'), 'w') as f_:
        for key in dict_page_pair:
            f_.write(str(key[0]))
            for value in dict_page_pair[key]:
                f_.write('\t' + str(value))
            f_.write('\n')


def write_file(cat_name, postfix):
    folder_cat = dict_cat2name[cat_name]
    f_train = open(join(folder_data, folder_cat, 'train.'+ postfix + '.txt'), 'w')
    if cat_name == 'm' or cat_name == 'p':
        f_test1 = open(join(folder_data, folder_cat, 'test.'+ postfix + '.txt.1'), 'w')
        f_test2 = open(join(folder_data, folder_cat, 'test.'+ postfix + '.txt.2'), 'w')
    line2task = []
    for line in file(join(folder_data, folder_cat, 'pair_info')):
        list_items = line.strip().split('\t')
        line2task.append(int(list_items[2]))
    line_id = 0
    for line in file(join(folder_data, folder_cat, 'pair_' + postfix)):
        if line2task[line_id] == 0:
            f_train.write(line)
        elif line2task[line_id] == 1:
            f_test1.write(line)
        else:
            f_test2.write(line)
        line_id += 1
    f_train.close()
    f_test1.close()
    f_test2.close()


if sys.argv[1] == 'load':
    load_pairs_folder()
elif sys.argv[1] == 'error':
    cat_name = sys.argv[2]
    begin = int(sys.argv[3])
    end = int(sys.argv[4])
    compute_error_rate(cat_name, begin, end)
elif sys.argv[1] == 'split':
    cat_name = sys.argv[2]
    post_fix = sys.argv[3]
    write_file(cat_name, post_fix)
