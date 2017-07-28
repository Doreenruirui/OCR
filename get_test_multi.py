from os.path import join, exists
from collections import OrderedDict
import os
import re
import sys
from PyLib.operate_file import load_obj, save_obj


def split_data(train_id, split_id):
    folder_test = join(folder_data, str(train_id))
    folder_train = join(folder_data, str(train_id), str(split_id))
    dict_test_id = load_obj(join(folder_test, 'split_' + str(train_id)))
    dict_id = load_obj(join(folder_train, 'split_' + str(split_id)))
    train_date = dict_id['train']
    dev_date = dict_id['test']
    test_date = dict_test_id['test']
    dict_split = OrderedDict()
    total_man = 0
    total_wit = 0
    total_man_wit = 0
    total_date = 0
    num_train = 0
    num_test = 0
    num_dev = 0
    num_other = 0
    lid = 0
    for line in file(join(folder_multi, 'pair.x.info')):
        items = line.strip('\n').split('\t')
        cur_begin = int(items[3])
        cur_end = int(items[4])
        line_id = int(items[1])
        cur_id = items[2]
        cur_date = re.findall('[0-9]{4}-[0-9]{2}-[0-9]{2}', cur_id)
        num_wit = int(items[5])
        num_manual = int(items[6])
        wit_line = -1
        if num_wit > 0:
            wit_line = int(items[7])
            total_wit += 1
        manual_line = -1
        if num_manual > 0:
            manual_line = int(items[8])
            total_man += 1
        print line_id
        if num_manual > 0 and num_wit > 0:
            if len(cur_date) > 0:
                if wit_line not in dict_split:
                    dict_split[wit_line] = []
                if cur_date[0] in train_date:
                    dict_split[wit_line].append((cur_begin, cur_end, line_id, manual_line, 0, total_date, cur_date))
                    num_train += 1
                elif cur_date[0] in test_date:
                    dict_split[wit_line].append((cur_begin, cur_end, line_id, manual_line, 1, total_date, cur_date))
                    num_test += 1
                elif cur_date[0] in dev_date:
                    dict_split[wit_line].append((cur_begin, cur_end,  line_id, manual_line, 2, total_date, cur_date))
                    num_dev += 1
                else:
                    dict_split[wit_line].append((cur_begin, cur_end,  line_id, manual_line, 3, total_date, cur_date))
                    num_other += 1
                total_date += 1
            total_man_wit += 1
        lid += 1
    folder_split = join(folder_multi, str(train_id) + '_' + str(split_id))
    if not exists(folder_split):
        os.makedirs(folder_split)
    save_obj(join(folder_split, 'split_' + str(train_id) + '_' + str(split_id)), dict_split)
    print num_train, num_test, num_dev, num_other, total_man_wit, total_man, total_wit, total_date



def write_data(train_id, split_id):
    folder_split = join(folder_multi, str(train_id) + '_' + str(split_id))
    if not exists(folder_split):
        os.makedirs(folder_split)
    dict_split = load_obj(join(folder_split, 'split_' + str(train_id) + '_' + str(split_id)))
    max_line = 0
    for item in dict_split:
        if dict_split[item][-1] > max_line:
            max_line = dict_split[item][-1]
    print max_line
    num_line = len(dict_split.keys())
    pair_x = []
    for line in file(join(folder_multi, 'pair.x')):
        pair_x.append(line.strip('\n'))
    pair_z = []
    for line in file(join(folder_multi, 'pair.z')):
        pair_z.append(line.strip('\n'))
    list_x = [None for _ in range(num_line)]
    list_y = [None for _ in range(num_line)]
    list_info = [None for _ in range(num_line)]
    line_id = 0
    print num_line, len(list_x), len(pair_x)
    for line in file(join(folder_multi, 'pair.y')):
        if line_id in dict_split:
            for info in dict_split[line_id]:
                b_id = info[0]
                e_id = info[1]
                x_id = info[2]
                z_id = info[3]
                flag_train = info[4]
                total_id = info[5]
                cur_date=info[6]
                print x_id
                list_x[total_id] = pair_x[x_id] + '\t' + line.strip('\n')
                list_y[total_id] = pair_z[z_id]
                list_info[total_id] = [cur_date, b_id, e_id, flag_train]
        line_id += 1

    list_file = {}
    for prefix in ['train', 'test', 'dev', 'other']:
        for postfix in ['x', 'y']:
            list_file[(prefix, postfix)] = open(join(folder_split, prefix + '.' + postfix + '.txt'), 'w')
        list_file[(prefix, 'info')] = open(join(folder_split, prefix + '.' + 'info.txt'), 'w')
    dict_id2train = {0: 'train', 1: 'test', 2: 'dev', 3: 'other'}
    for i in range(num_line):
        cur_x = list_x[i]
        cur_y = list_y[i]
        cur_info = list_info[i]
        print cur_info
        if '#' in cur_y:
            continue
        else:
            cur_train = dict_id2train[cur_info[-1]]
            list_file[(cur_train, 'x')].write(cur_x + '\n')
            list_file[(cur_train, 'y')].write(cur_y + '\n')
            list_file[(cur_train, 'info')].write('\t'.join(map(str, cur_info[:-1])) + '\n')

    for item in list_file:
        list_file[item].close()





folder_data = '/scratch/dong.r/Dataset/OCR/richmond'
folder_multi = '/scratch/dong.r/Dataset/OCR/multi_new'

train_ratio = 0.8
tid = 0
sid = 0
split_data(tid, sid)
write_data(tid, sid)
