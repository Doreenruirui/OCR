from os.path import join, exists
import json
from PyLib.operate_file import save_obj, load_obj
from levenshtein import align, align_pair
from multiprocessing import Pool
import os


folder_data = '/scratch/dong.r/Dataset/OCR/data'


def error_rate(P, nthread, flag_char, list_x, list_y, len_y):
    dis_xy = align_pair(P, list_x, list_y, nthread, flag_char=flag_char)
    res = []
    for i in range(len(dis_xy)):
        res.append(dis_xy[i] * 1. / len_y[i])
    return res


def load_pairs_all():
    keys = ['b1', 'e1', 'b2', 'e2']
    info_file = open(join(folder_data, 'pair_info'), 'w')
    pair_file = open(join(folder_data, 'pairs'), 'w')
    num_pair = 0
    list_str1 = []
    list_str2 = []
    for line in file(join(folder_data, 'pairs-n5.json')):
        cur_dict = json.loads(line.strip())
        num_pair += len(cur_dict['pairs'])
        if len(cur_dict['pairs']) > 2:
            for pair in cur_dict['pairs']:
                str1 = pair['_1'].strip().replace('\n', ' ')
                str2 = pair['_2'].strip().replace('\n', ' ')
                if len(str1) > 0 and len(str2) > 0:
                    pair_file.write(str1.encode('utf-8') + '\t\t' + str2.encode('utf-8') + '\n')
                    cur_info = cur_dict['id1'][:12] + '\t' + '\t'.join(map(str, [cur_dict[ele] for ele in keys]))
                    info_file.write(cur_info + '\n')
                    list_str1.append(str1.encode('utf-8'))
                    list_str2.append(str2.encode('utf-8'))
    info_file.close()
    pair_file.close()
    pool = Pool(45)
    len_y = [len(ele) for ele in list_str1]
    dis_xy = error_rate(pool, 45, 1, list_str1, list_str2, len_y)
    dis_file = open(join(folder_data, 'pairs_error'), 'w')
    for ele in dis_xy:
        dis_file.write('\t%.5f\n' % ele)
    dis_file.close()
    print (num_pair)


load_pairs_all()
