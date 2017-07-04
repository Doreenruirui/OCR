from PyLib import operate_file as opf
from PyLib.operate_file import save_obj, load_obj
from levenshtein import align_pair
from my_utils import error_rate_line, split_with_ratio
from collections import OrderedDict
from os.path import join, exists
import  numpy as np
import json
import os
import re
import sys
import random


dict_cat = {'monograph', 'periodical'}
dict_cat2name = {'m': 'monograph', 'p': 'periodical', 'r': 'richmond'}


def load_pair_info():
    if cat_name == 'richmond':
        pair_info = []
        for line in file(join(folder_data, 'pair_info')):
            date = line.split('\t')[0][:-2]
            pair_info.append(date)
    else:
        pair_info = np.loadtxt(join(folder_data,
                                    'pair_info'), dtype=int)[:, 0]
    return pair_info


def load_pair_start_end():
    if cat_name == 'richmond':
        begin_x = []
        end_x = []
        begin_y = []
        end_y = []
        date = []
        for line in file(join(folder_data, 'pair_info')):
            cur_line = line.strip().split('\t')
            begin_x.append(int(cur_line[3]))
            end_x.append(int(cur_line[4]))
            begin_y.append(int(cur_line[1]))
            end_y.append(int(cur_line[2]))
            date.append(cur_line[0])
    else:
        begin_x = []
        end_x = []
        date = []
        for line in file(join(folder_data, 'pair_info')):
            cur_line = line.strip().split('\t')
            date.append(cur_line[0])
            begin_x.append(int(cur_line[1]))
            end_x.append(int(cur_line[2]))
        begin_y = begin_x
        end_y = end_x
    return date, begin_x, end_x, begin_y, end_y


def load_pair_dis():
    pair_dis = np.loadtxt(join(folder_data, 'pair_error'))
    return pair_dis


def load_pairs_rich():
    folder_origin_data = '/scratch/dong.r/Dataset/unprocessed/richmond'
    keys = ['b1', 'e1', 'b2', 'e2']
    info_file = open(join(folder_data, 'pair_info'), 'w')
    pair_file_x = open(join(folder_data, 'pair_x'), 'w')
    pair_file_y = open(join(folder_data, 'pair_y'), 'w')
    pair_file_z = open(join(folder_data, 'pair_z'), 'w')
    for line in file(join(folder_origin_data, 'pairs-n5.json')):
        cur_dict = json.loads(line.strip())
        for pair in cur_dict['pairs']:
            str1 = re.sub('\n', ' ', pair['_1'].strip()).strip()
            str2 = re.sub('\n', ' ', pair['_2'].strip()).strip()
            str3 = ' '.join([ele for ele in re.sub('\n', ' ', pair['_2']).split() if len(ele.strip()) > 0])
            pair_file_y.write(str1.encode('utf-8') + '\n')
            pair_file_x.write(str2.encode('utf-8') + '\n')
            pair_file_z.write(str3.encode('utf-8') + '\n')
            cur_info = cur_dict['id1'][:12] + '\t' + '\t'.join(map(str, [cur_dict[ele] for ele in keys]))
            info_file.write(cur_info + '\n')
    info_file.close()
    pair_file_x.close()
    pair_file_y.close()
    pair_file_z.close()


def load_pairs_eng(chunk_size):
    folder_origin_data = '/scratch/dong.r/Dataset/unprocessed/ICDAR2017'
    cn = cat_name
    if not exists(folder_data):
        os.makedirs(folder_data)
    f_info = open(join(folder_data, 'pair_info'), 'w')
    f_x = open(join(folder_data, 'pair_x'), 'w')
    f_y = open(join(folder_data, 'pair_y'), 'w')
    f_z = open(join(folder_data, 'pair_z'), 'w')
    files = [f for f in os.listdir(join(folder_origin_data,
                                        'eng_' + cn))]
    for fn in files:
        full_name = join(folder_origin_data,
                         'eng_' + cn, fn)
        if not fn.startswith('.') and os.path.isfile(full_name):
            with open(full_name, 'r') as f_:
                lines = f_.readlines()
                ocr = lines[1].translate(None, '\r\n').decode('utf-8')
                ocr = ocr[len('[OCR_aligned]') + 1:]
                gs = lines[2].translate(None, '\r\n').decode('utf-8')
                gs = gs[len('[ GS_aligned]') + 1:]
                len_o = len(ocr)
                len_g = len(gs)
                if len_o != len_g:
                    raise 'OCR and Gold Standard not Aligned!'
                id_char = 0
                while id_char < len_o:
                    cur_len = min(int(np.floor(random.random() * 10)) + 40, len_o - id_char)
                    begin = id_char
                    end = begin + cur_len
                    cur_ocr = re.sub('@|#', '', ocr[begin:end]).strip()
                    cur_gs = re.sub('@|#', '', gs[begin:end]).strip()
                    cur_gs_orig = re.sub('@|#', '', gs[begin:end])
                    f_x.write(cur_ocr.encode('utf-8') + '\n')
                    f_y.write(cur_gs.encode('utf-8') + '\n')
                    f_z.write(cur_gs_orig.encode('utf-8') + '\n')
                    id_char = end
                    f_info.write('%d\t%d\t%d\n' % (int(fn[:-len('.txt')]), begin, end))
    f_info.close()
    f_x.close()
    f_y.close()
    f_z.close()
    for task in ['task1', 'task2']:
        folder_in = join(folder_origin_data,
                         'eng_' + cn + '_' + task)
        files = [f for f in os.listdir(folder_in)]
        f_info = open(join(folder_data, task + '_info'), 'w')
        f_x = open(join(folder_data, task + '_x'), 'w')
        for fn in files:
            full_name = join(folder_in, fn)
            if not fn.startswith('.') and os.path.isfile(full_name):
                with open(full_name, 'r') as f_:
                    lines = f_.readlines()
                    ocr = lines[0].translate(None, '\r\n').decode('utf-8')
                    ocr = ocr[len('[OCR_toInput]') + 1:]
                    len_o = len(ocr)
                    id_char = 0
                    while id_char < len_o:
                        cur_len = min(int(np.floor(random.random() * 10)) + 40, len_o - id_char)
                        begin = id_char
                        end = begin + cur_len
                        cur_ocr = ocr[begin:end]
                        f_x.write(cur_ocr.encode('utf-8') + '\n')
                        f_info.write('%d\t%d\t%d\n' % (int(fn[:-len('.txt')]), begin, end))
                        id_char = end
        f_info.close()
        f_x.close()


def compute_error_rate():
    folder_name = folder_data
    file_ocr = join(folder_name, 'pair_x')
    file_truth = join(folder_name, 'pair_y')
    file_error = join(folder_name, 'pair_error')
    error_rate_line(file_error, file_ocr, file_truth, 0, -1)


def split_train_test(train_ratio, split_id):
    folder_split = join(folder_data, str(split_id))
    if not exists(folder_split):
        os.makedirs(folder_split)
    pair_info = load_pair_info()
    all_ele = OrderedDict()
    for ele in pair_info:
        if ele not in all_ele:
            all_ele[ele] = 1
    date_list = all_ele.keys()
    num_file = len(date_list)
    index_train, index_test = split_with_ratio(num_file, train_ratio)
    train_id = np.sort(index_train)
    test_id = np.sort(index_test)
    train_date = OrderedDict()
    for index in train_id:
        train_date[date_list[index]] = 1
    test_date = OrderedDict()
    for index in test_id:
        test_date[date_list[index]] = 1
    save_obj(join(folder_split, 'split_' + str(split_id)),
             {'train': train_date, 'test': test_date})


def split_train_dev(train_ratio, train_id, split_id):
    folder_train = join(folder_data, str(train_id))
    folder_split = join(folder_data, str(train_id), str(split_id))
    if not exists(folder_split):
        os.makedirs(folder_split)
    dict_id = load_obj(join(folder_train,
                            'split_' + str(train_id)))
    train_date = dict_id['train'].keys()
    num_file = len(train_date)
    index_train, index_test = split_with_ratio(num_file, train_ratio)
    train_id = np.sort(index_train)
    test_id = np.sort(index_test)
    new_train_date = OrderedDict()
    for index in train_id:
        new_train_date[train_date[index]] = 1
    test_date = OrderedDict()
    for index in test_id:
        test_date[train_date[index]] = 1
    save_obj(join(folder_split, 'split_' + str(split_id)),
             {'train': new_train_date, 'test': test_date})


def write_file_train_test(split_id):
    folder_train = join(folder_data, str(split_id))

    def write_file(prefix, data_id, post_fix):
        with open(join(folder_train, prefix + '.' + post_fix + '.txt'), 'w') as f_:
            for cur_id in data_id:
                if post_fix == 'x':
                    f_.write(x_list[cur_id] + '\n')
                else:
                    f_.write(y_list[cur_id] + '\n')
    dict_id = load_obj(join(folder_train, 'split_' + str(split_id)))
    test_date = dict_id['test']
    pairs_info = load_pair_info()
    x_list = []
    for line in file(join(folder_data, 'pair_x')):
        x_list.append(line.strip())
    y_list = []
    for line in file(join(folder_data, 'pair_y')):
        y_list.append(line.strip())
    z_list = []
    for line in file(join(folder_data, 'pair_z')):
        z_list.append(line.strip())
    num_pair = len(pairs_info)
    test_id = OrderedDict()
    for i in range(num_pair):
        if 'UNCLEAR' in y_list[i]:
            continue
        cur_date = pairs_info[i]
        if len(y_list[i]) > 0:
            if cur_date in test_date:
                test_id[i] = 1
    print len(test_id)
    print len(x_list)
    print len(y_list)
    write_file('test', test_id, 'x')
    write_file('test', test_id, 'y')
    write_file('test', test_id, 'z')
    f_ = open(join(folder_train, 'test.id'), 'w')
    for cur_id in test_id:
        f_.write('%d\n' % cur_id)
    f_.close()


def write_file_train_dev(error_ratio, train_id, split_id):
    folder_train = join(folder_data, str(train_id), str(split_id))
    folder_out = join(folder_train, str(error_ratio))
    if not exists(folder_out):
        os.makedirs(folder_out)
    def write_file(prefix, data_id, post_fix):
        with open(join(folder_out, prefix + '.' + post_fix + '.txt'), 'w') as f_:
            for cur_id in data_id:
                if post_fix == 'x':
                    f_.write(x_list[cur_id] + '\n')
                else:
                    f_.write(y_list[cur_id] + '\n')
    dict_id = load_obj(join(folder_train, 'split_' + str(split_id)))
    train_date = dict_id['train']
    dev_date = dict_id['test']
    pairs_info = load_pair_info()
    pair_dis = load_pair_dis()
    x_list = []
    for line in file(join(folder_data, 'pair_x')):
        x_list.append(line.strip())
    y_list = []
    for line in file(join(folder_data, 'pair_y')):
        y_list.append(line.strip())
    z_list = []
    for line in file(join(folder_data, 'pair_z')):
        z_list.append(line.strip())
    num_pair = len(pairs_info)
    train_id = OrderedDict()
    dev_id = OrderedDict()
    for i in range(num_pair):
        if 'UNCLEAR' in y_list[i]:
            continue
        cur_date = pairs_info[i]
        if len(y_list[i]) > 0:
            if cur_date in train_date:
                if len(x_list[i]) > 0 and pair_dis[i][0] * 1. / pair_dis[i][2] < 0.01 * error_ratio:
                    train_id[i] = 1
            elif cur_date in dev_date:
                dev_id[i] = 1
    print len(train_id), len(dev_id)
    print len(x_list)
    print len(y_list)
    write_file('train', train_id, 'x')
    write_file('train', train_id, 'y')
    write_file('dev', dev_id, 'x')
    write_file('dev', dev_id, 'y')
    write_file('dev', dev_id, 'z')
    f_ = open(join(folder_train, 'train.id'), 'w')
    for cur_id in train_id:
        f_.write('%d\n' % cur_id)
    f_.close()
    f_ = open(join(folder_train, 'dev.id'), 'w')
    for cur_id in train_id:
        f_.write('%d\n' % cur_id)
    f_.close()


def write_file_train_test_lm(split_id):
    folder_train = join(folder_data, str(split_id))
    dict_id = load_obj(join(folder_train, 'split_' + str(split_id)))
    test_date = dict_id['test']
    date, bx, ex, by, ey = load_pair_start_end()
    date_info = load_pair_info()
    dict_group2line = OrderedDict()
    num_pair = len(date)
    for i in range(num_pair):
        cur_date = date_info[i]
        if cur_date in test_date:
            true_date = date[i]
            if true_date not in dict_group2line:
                dict_group2line[true_date] = OrderedDict()
            cur_b = bx[i]
            if cur_b not in dict_group2line[true_date]:
                dict_group2line[true_date][cur_b] = [ex[i]]
            dict_group2line[true_date][cur_b].append(i)
    # merge continuous groups
    for date in dict_group2line:
        cur_group = dict_group2line[date]
        for cur_b in cur_group:
            if len(cur_group[cur_b]) > 0:
                cur_e = cur_group[cur_b][0]
                if cur_e in cur_group:
                    print len(cur_group[cur_b]) - 1, len(cur_group[cur_e]) - 1
                    dict_group2line[date][cur_b] += dict_group2line[date][cur_e]
                    dict_group2line[date][cur_b][0] = dict_group2line[date][cur_e][0]
                    dict_group2line[date][cur_e] = []
    save_obj(join(folder_train, 'test.group'), dict_group2line)


def write_file_train_dev_lm(error_ratio, train_id, split_id):
    folder_train = join(folder_data, str(train_id), str(split_id))
    folder_out = join(folder_train, str(error_ratio))
    if not exists(folder_out):
        os.makedirs(folder_out)
    dict_id = load_obj(join(folder_train, 'split_' + str(split_id)))
    dev_date = dict_id['test']
    date, bx, ex, by, ey = load_pair_start_end()
    date_info = load_pair_info()
    num_pair = len(date)
    dict_group2line = OrderedDict()
    for i in range(num_pair):
        cur_date = date_info[i]
        if cur_date in dev_date:
            true_date = date[i]
            if true_date not in dict_group2line:
                dict_group2line[true_date] = OrderedDict()
            cur_b = bx[i]
            if cur_b not in dict_group2line[true_date]:
                dict_group2line[true_date][cur_b] = [ex[i]]
            dict_group2line[true_date][cur_b].append(i)
    for date in dict_group2line:
        cur_group = dict_group2line[date]
        for cur_b in cur_group:
            if len(cur_group[cur_b]) > 0:
                cur_e = cur_group[cur_b][0]
                if cur_e in cur_group:
                    dict_group2line[date][cur_b] += dict_group2line[date][cur_e]
                    dict_group2line[date][cur_b][0] = dict_group2line[date][cur_e][0]
                    dict_group2line[date][cur_e] = []
    save_obj(join(folder_train, 'dev.group'), dict_group2line)


cat_name = dict_cat2name[sys.argv[1]]
folder_data = join('/scratch/dong.r/Dataset/OCR/', cat_name)
if sys.argv[2] == 'load':
    if sys.argv[1] == 'r':
        load_pairs_rich()
    else:
        load_pairs_eng(40)
elif sys.argv[2] == 'error':
    compute_error_rate()
elif sys.argv[2] == 'split':
    if sys.argv[3] == 'test':
        train_ratio = 0.8
        sid = int(sys.argv[4])
        split_train_test(train_ratio, sid)
    if sys.argv[3] == 'dev':
        train_ratio = 0.8
        tid = int(sys.argv[4])
        sid = int(sys.argv[5])
        split_train_dev(train_ratio, tid, sid)
elif sys.argv[2] == 'write':
    if sys.argv[3] == 'test':
        sid = int(sys.argv[4])
        write_file_train_test(sid)
    if sys.argv[3] == 'dev':
        tid = int(sys.argv[4])
        sid = int(sys.argv[5])
        ratio = int(sys.argv[6])
        write_file_train_dev(ratio, tid, sid)
elif sys.argv[2] == 'group':
    if sys.argv[3] == 'test':
        sid = int(sys.argv[4])
        write_file_train_test_lm(sid)
    if sys.argv[3] == 'dev':
        tid = int(sys.argv[4])
        sid = int(sys.argv[5])
        ratio = int(sys.argv[6])
        write_file_train_dev_lm(ratio, tid, sid)



