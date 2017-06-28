from os.path import join
import json
from levenshtein import align_pair
from multiprocessing import Pool
import os
import re
from collections import OrderedDict
from PyLib import operate_file as opf
import  numpy as np


def error_rate(P, nthread, flag_char, list_x, list_y, len_y):
    dis_xy = align_pair(P, list_x, list_y, nthread, flag_char=flag_char)
    res = []
    for i in range(len(dis_xy)):
        res.append(dis_xy[i] * 1. / len_y[i])
    return res


def load_pairs_folder():
    keys = ['b1', 'e1', 'b2', 'e2']
    folder_data = '/scratch/dong.r/Dataset/OCR/eng'
    info_file = open(join(folder_data, 'pair_info'), 'w')
    pair_file = open(join(folder_data, 'pairs'), 'w')
    num_pair = 0
    list_str1 = []
    list_str2 = []
    files = [f for f in os.listdir(join(folder_data, 'pass.json'))]
    files = [ele for ele in files if os.path.isfile(join(folder_data, 'pass.json', ele))]
    pair_id = 0
    line_id = 0
    for fname in files:
        if fname.startswith('part-'):
            for line in file(join(folder_data, 'pass.json', fname)):
                cur_dict = json.loads(line.strip())
                num_pair += len(cur_dict['pairs'])
                for pair in cur_dict['pairs']:
                    str1 = re.sub('\n', '<NEWLINE>', pair['_1'])
                    str2 = re.sub('\n', '<NEWLINE>', pair['_2'])
                    pair_file.write(str1.encode('utf-8') + '\t\t' + str2.encode('utf-8') + '\n')
                    cur_info = cur_dict['id1'] + '\t' + cur_dict['id2'] + '\t' + '\t'.join(map(str, [cur_dict[ele] for ele in keys]))
                    info_file.write(cur_info + '\n')
                    list_str1.append(str1.encode('utf-8'))
                    list_str2.append(str2.encode('utf-8'))
                    pair_id += 1
                line_id += 1
    print line_id
    print pair_id
    info_file.close()
    pair_file.close()
    pool = Pool(50)
    len_y = [len(ele) for ele in list_str1]
    dis_xy = error_rate(pool, 50, 1, list_str1, list_str2, len_y)
    dis_file = open(join(folder_data, 'pairs_error'), 'w')
    for ele in dis_xy:
        dis_file.write('\t%.5f\n' % ele)
    dis_file.close()
    print (num_pair)


def get_train_data():
    folder_data = '/scratch/dong.r/Dataset/OCR/eng'
    dict_page_pair = OrderedDict()
    line_id = 0
    line2task = []
    for line in file(join(folder_data, 'pair_info')):
        list_items = line.strip().split('\t')
        task = list_items[1]
        b1 = int(list_items[2])
        b2 = int(list_items[3])
        key = (b1, b2)
        if '_task1' in task:
            line2task.append(1)
        elif '_task2' in task:
            line2task.append(2)
        else:
            line2task.append(0)
        if key not in dict_page_pair:
            dict_page_pair[key] = []
        dict_page_pair[key].append(line_id)
        line_id += 1
    with open(join(folder_data, 'pairs_date'), 'w') as f_:
        for key in dict_page_pair:
            f_.write(str(key[0]) + '\t' + str(key[1]))
            for value in dict_page_pair[key]:
                f_.write('\t' + str(value))
            f_.write('\n')
    np.save(join(folder_data, 'line2task'), np.asarray(line2task))


def write_file():
    folder_data = '/scratch/dong.r/Dataset/OCR/eng'
    fx_train = open(join(folder_data, 'train.x.txt'), 'w')
    fy_train = open(join(folder_data, 'train.y.txt'), 'w')
    fx_test1 = open(join(folder_data, 'test.x.txt.1'), 'w')
    fy_test1 = open(join(folder_data, 'test.y.txt.1'), 'w')
    fx_test2 = open(join(folder_data, 'test.x.txt.2'), 'w')
    fy_test2 = open(join(folder_data, 'test.y.txt.2'), 'w')
    line2task = np.load(join(folder_data, 'line2task.npy'))
    line_id = 0
    for pair in file(join(folder_data, 'pairs')):
        y, x = pair[-1].split('\t')
        if line2task[line_id] == 0:
            fx_train.write(x + '\n')
            fy_train.write(y + '\n')
        elif line2task[line_id] == 1:
            fx_test1.write(x + '\n')
            fy_test1.write(y + '\n')
        else:
            fx_test2.write(x + '\n')
            fy_test2.write(y + '\n')
    fx_train.close()
    fy_train.close()
    fx_test1.close()
    fy_test1.close()
    fx_test2.close()
    fy_test2.close()


# load_pairs_folder()
get_train_data()
write_file()