from os.path import join, exists
import sys
from collections import OrderedDict


folder_multi = '/gss_gpfs_scratch/dong.r/Dataset/OCR/multi'


def get_train_data(folder_data, train):
    dict_data_seq = OrderedDict()
    line_id = 0
    last_end = 0
    num_line = 0
    for line in file(join(folder_data, train + '.info.txt')):
        cur_info = line.strip('\n').split('\t')
        key1 = (cur_info[0], cur_info[1])
        start = cur_info[2]
        end = cur_info[3]
        if key1 not in dict_data_seq:
            dict_data_seq[key1] = []
            dict_data_seq[key1].append([line_id])
        else:
            if start == last_end:
                dict_data_seq[key1][-1].append(line_id)
            else:
                dict_data_seq[key1].append([line_id])
        last_end = end
        num_line += 1
        line_id += 1
    group_id = 0
    line2group = [-1 for _ in range(num_line)]
    for key in dict_data_seq:
        for group in dict_data_seq[key]:
            for line_id in group:
                line2group[line_id] = group_id
            group_id += 1
    with open(join(folder_data, train + '.line2group'), 'w') as f_:
        for item in line2group:
            f_.write(str(item) + '\n')


folder_test = join(folder_multi, sys.argv[1])
folder_valid = join(folder_multi, sys.argv[1], sys.argv[2])
get_train_data(folder_test, 'man_wit.test')
get_train_data(folder_valid, 'man_wit.dev')
get_train_data(folder_test, 'man.test')
get_train_data(folder_valid, 'man.dev')