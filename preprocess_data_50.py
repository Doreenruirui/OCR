from os.path import join, exists
import json
import numpy as np
from PyLib.operate_file import save_obj
from levenshtein import align
import os


folder_data = '/scratch/dong.r/Dataset/OCR/data'

def load_pairs():
    keys = ['b1', 'e1', 'b2', 'e2']
    info_file = open(join(folder_data, 'pair_info_50'), 'w')
    pair_file = open(join(folder_data, 'pairs_50'), 'w')
    num_pair = 0
    for line in file(join(folder_data, 'pairs-n5.json')):
        cur_dict = json.loads(line.strip())
        num_pair += len(cur_dict['pairs'])
        if len(cur_dict['pairs']) > 2:
            for pair in cur_dict['pairs'][1:-1]:
                str1 = pair['_1'].strip().replace('\n', ' ')
                str2 = pair['_2'].strip().replace('\n', ' ')
                if '###UNCLEAR###' in str1:
                    continue
                if len(str1) > 0 and len(str2) > 0:
                    if str2[-1] == '-' and str2[-2] != '-':
                        str2 = str2[:-1]
                    dis = align(str1, str2)
                    if dis < 0.5 * len(str2):
                        # print dis
                        pair_file.write(str1.encode('utf-8') + '\t\t' + str2.encode('utf-8') + '\n')
                        cur_info = cur_dict['id1'][:12] + '\t' + \
                                   '\t'.join(map(str, [cur_dict[ele] for ele in keys]))
                        info_file.write(cur_info + '\n')
    info_file.close()
    pair_file.close()
    print (num_pair)

def split_dataset(train_ratio):
    if not exists(join(folder_data, 'char_50')):
        os.makedirs(join(folder_data, 'char_50'))
    def write_file(prefix, data_id):
        with open(join(folder_data, 'char_50', prefix + '.x.txt'), 'w') as f_:
            for cur_id in data_id:
                f_.write(x_list[cur_id] + '\n')
        with open(join(folder_data, 'char_50', prefix + '.y.txt'), 'w') as f_:
            for cur_id in data_id:
                f_.write(y_list[cur_id] + '\n')
    with open(join(folder_data, 'pairs_50'), 'r') as f_:
        num_data = len(f_.readlines())
    num_train = int(np.floor(num_data * train_ratio))
    num_test = num_data - num_train
    num_dev = num_train - int(np.floor(num_train * train_ratio))
    num_train = num_data - num_test - num_dev
    rand_index = np.arange(num_data)
    np.random.shuffle(rand_index)
    train_id = np.sort(rand_index[:num_train])
    dev_id = np.sort(rand_index[num_train: num_train + num_dev])
    test_id = np.sort(rand_index[-num_test:])
    save_obj(join(folder_data, 'pair_split_50'),  {'train': train_id,
                                                'dev': dev_id,
                                                'test': test_id})
    x_list = []
    y_list = []
    for line in file(join(folder_data, 'pairs_50')):
        y, x = line.strip().split('\t\t')
        if len(x) > 0 and len(y) > 0:
            x_list.append(x)
            y_list.append(y)
    print len(x_list)
    print len(y_list)
    write_file('train', train_id)
    write_file('test', test_id)
    write_file('dev', dev_id)



#load_pairs()
split_dataset(train_ratio=0.8)
