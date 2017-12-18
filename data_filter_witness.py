from os.path import join, exists
import os
import sys
import re
from multiprocessing import Pool
from levenshtein import align_pair


folder_data = '/scratch/dong.r/Dataset/OCR'
arg_folder = sys.argv[1]
arg_train = sys.argv[2]
arg_out = sys.argv[3]


def remove(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def process_lines(folder_name, prefix='dev'):
    global folder_data
    cur_folder = join(folder_data, folder_name)
    with open(join(cur_folder, prefix + '.x.txt'), 'r') as f_:
        lines = f_.readlines()
    with open(join(cur_folder, prefix + '.x.txt'), 'w') as f_:
        for line in lines:
            items = line.strip().split('\t')
            items = [remove(ele).strip() for ele in items]
            items = [ele for ele in items if len(ele) > 0]
            new_items = []
            for ele in items:
                first_index = 0
                while first_index < len(ele):
                    if ele[first_index].isalnum():
                        break
                    first_index += 1
                last_index = len(ele) - 1
                while last_index >= 0:
                    if ele[last_index].isalnum() or ele[last_index] in {'.': 0,
                                                            ',': 0,
                                                            '?': 0,
                                                            '!': 0,
                                                            ';': 0,
                                                            '"': 0,
                                                            '\'': 0,
                                                            ':': 0}:
                        break
                    last_index -= 1
                cur_ele = ele[first_index:last_index]
                if len(cur_ele) > 0:
                    new_items.append(cur_ele)
            f_.write('\t'.join(new_items) + '\n')


def evaluate_distance(folder_name, prefix='dev'):
    global folder_data
    cur_folder = join(folder_data, folder_name)
    list_x = []
    num = []
    list_y = []
    with open(join(cur_folder, prefix + '.x.txt'), 'r') as f_:
        line_id = 0
        for line in f_.readlines():
            cur_line = remove(line).lower().strip('\n').split('\t')
            cur_line = [ele.strip() for ele in cur_line if len(ele.strip()) > 0]
            list_x += cur_line
            num.append(len(cur_line))
            list_y += [cur_line[0] for _ in cur_line[1:]]
            line_id += 1
    pool = Pool(100)
    dis_xy = align_pair(pool, list_x, list_y, flag_char=1, flag_low=1)
    line_id = 0
    with open(join(cur_folder, prefix + '.xec.txt'), 'w') as f_:
        for i in range(len(list_y)):
            new_line_id = line_id + num[i]
            cur_dis = dis_xy[line_id: new_line_id]
            f_.write('\t'.join(map(str, cur_dis)) + '\t' + str(len(list_y[i])) + '\n')
            line_id = new_line_id


def filter_witness_distance(folder_in, prefix, folder_out):
    folder_train = join(folder_data, folder_in)
    out_folder = join(folder_data, folder_out)

    list_x = []
    list_index = []
    distance = []

    if not exists(folder_out):
        os.makedirs(folder_out)

    for line in file(join(folder_train, arg_train + '.x.txt')):
        items = line.strip('\n').split('\t')


def evaluate_length(folder_name, prefix='dev'):
    global folder_data
    cur_folder = join(folder_data, folder_name)
    len_x = []
    with open(join(cur_folder, prefix + '.x.txt'), 'r') as f_:
        line_id = 0
        for line in f_.readlines():
            cur_line = remove(line).lower().strip('\n').split('\t')
            cur_line = [ele.strip() for ele in cur_line]
            len_x.append([len(ele) for ele in cur_line])
            line_id += 1
    with open(join(cur_folder, prefix + '.len.txt'), 'w') as f_:
        for ele in len_x:
            f_.write('\t'.join(map(str, ele))+ '\n')


def filter_witness_length(folder_in, prefix, folder_out):
    folder_train = join(folder_data, folder_in)
    out_folder = join(folder_data, folder_out)
    len_x = []
    if not exists(folder_out):
        os.makedirs(folder_out)
    for line in file(join(folder_train, prefix + '.len.txt')):
        items = line.strip('\n').split('\t')
        if len(items) == 1 and len(items[0]) == 0:
            len_x.append([])
        else:
            len_x.append(map(float, items))
    avg_wit = 0
    num_empty = 0
    filter_avg_wit = 0
    line_id = 0
    num_line = 0
    remain_index = []
    for line in len_x:
        line_id += 1
        print line_id
        if max(line) == 0:
            remain_index.append([])
        else:
            num_line += 1
            avg_wit += len(line) - 1
            dict_len = {}
            for ele in line:
                dict_len[ele] = dict_len.get(ele, 0) + 1
            if len(line) == 1:
                min_dis = 0
                remain_index.append([0])
            else:
                min_dis = min([abs(ele - line[0]) for ele in line[1:]])
                if min_dis <= 1:
                    cur_num = 0
                    for ele in dict_len:
                        if abs(ele - line[0]) <= 1:
                            cur_num += dict_len[ele]
                    filter_avg_wit += cur_num - 1
                    cur_index = []
                    for j in range(len(line)):
                        if abs(line[j] - ele) <= 1:
                            cur_index.append(j)
                    remain_index.append(cur_index)
                elif min_dis <= 2:
                    cur_num = 0
                    for ele in dict_len:
                        if abs(ele - line[0]) <= 2:
                            cur_num += dict_len[ele]
                    filter_avg_wit += cur_num - 1
                    cur_index = []
                    for j in range(len(line)):
                        if abs(line[j] - ele) <= 2:
                            cur_index.append(j)
                    remain_index.append(cur_index)
                elif min_dis <= 3:
                    cur_num = 0
                    for ele in dict_len:
                        if abs(ele - line[0]) <= 3:
                            cur_num += dict_len[ele]
                    filter_avg_wit += cur_num - 1
                    cur_index = []
                    for j in range(len(line)):
                        if abs(line[j] - ele) <= 3:
                            cur_index.append(j)
                    remain_index.append(cur_index)
                else:
                    num_empty += 1
                    remain_index.append([0])
    with open(join(folder_train, prefix + '.x.txt'), 'r') as f_:
        lines = f_.readlines()
    with open(join(out_folder, prefix + '.x.txt'), 'w') as f_:
        line_id = 0
        for line in lines:
            items = line.strip().split('\t')
            if len(remain_index[line_id]) > 0:
                f_.write('\t'.join([items[j] for j in remain_index[line_id]]) + '\n')
            else:
                f_.write('\n')
            line_id += 1


    # print num_empty
    # print avg_wit / num_line
    # print filter_avg_wit / num_line


#process_lines(arg_folder, arg_train)
evaluate_length(arg_folder, arg_train)
filter_witness_length(arg_folder, arg_train, arg_out)
