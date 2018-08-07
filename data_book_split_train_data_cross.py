from os.path import join, exists
import os
from PyLib.operate_file import load_obj, save_obj
from levenshtein import align_pair
from multiprocessing import Pool
from collections import OrderedDict
from util_my import split_with_ratio
import numpy as np
import sys
import re


def compute_error_rate(train_id, split_id):
    folder_train = join(folder_multi, str(train_id), str(split_id))
    list_x = []
    list_y = []
    for line in file(join(folder_train, 'man.train.x.txt')):
        list_x.append(line.strip('\n'))
    for line in file(join(folder_train, 'man_wit.train.x.txt')):
        list_x.append(line.strip('\n').split('\t')[0])
    for line in file(join(folder_train, 'man.train.y.txt')):
        list_y.append(line.strip() + '\n')
    for line in file(join(folder_train, 'man_wit.train.y.txt')):
        list_y.append(line.strip() + '\n')
    pool = Pool(100)
    dis = align_pair(pool, list_x, list_y)
    np.savetxt(join(folder_train, 'distance'), dis, fmt='%d')


def get_rand_train_dev(train_id):
    folder_train = join(folder_multi, str(train_id))
    dict_id = load_obj(join(folder_train, 'split_' + str(train_id)))
    train_date = dict_id['train'].keys()
    num_date = len(train_date)
    rand_index = np.arange(num_date)
    np.random.shuffle(rand_index)
    np.savetxt(join(folder_train, 'date_random'), rand_index, fmt='%d')


def split_train_test(num_fold, split_id):
    dict_date = load_obj(join(folder_multi, 'date_info'))
    date_list = dict_date.keys()
    num_date = len(date_list)
    size_fold = int(np.floor(num_date * 1. / num_fold))
    rand_index = np.loadtxt(join(folder_multi, 'date_random'), dtype=int)
    folder_split = join(folder_multi, str(split_id))
    if not exists(folder_split):
        os.makedirs(folder_split)
    if split_id < num_fold - 1:
        start = split_id * size_fold
        end = (split_id + 1) * size_fold
    else:
        start = split_id * size_fold
        end = num_date
    index_test = rand_index[start:end]
    index_train = np.concatenate((rand_index[:start], rand_index[end:]))
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


def split_train_dev(num_fold, train_id, split_id):
    folder_train = join(folder_multi, str(train_id))
    dict_id = load_obj(join(folder_train,
                            'split_' + str(train_id)))
    train_date = dict_id['train'].keys()
    num_date = len(train_date)
    size_fold = int(np.floor(num_date * 1. / num_fold))
    rand_index = np.loadtxt(join(folder_train, 'date_random'), dtype=int)
    folder_split = join(folder_multi, str(train_id), str(split_id))
    if not exists(folder_split):
        os.makedirs(folder_split)
    if split_id < num_fold - 1:
        start = split_id * size_fold
        end = (split_id + 1) * size_fold
    else:
        start = split_id * size_fold
        end = num_date
    index_test = rand_index[start:end]
    index_train = np.concatenate((rand_index[:start], rand_index[end:]))
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


def split_date(train_id, split_id):

    def get_index(name_train, name_man):
        cur_index = {}
        line_id = 0
        for line in file(join(folder_multi, name_man + '.info.txt')):
            items = line.split('\t')
            cur_date = items[0]
            if name_train == 'train':
                if cur_date not in dict_date['test'] and cur_date not in dict_date['dev']:
                    cur_index[line_id] = 1
            else:
                if cur_date in dict_date[name_train]:
                    cur_index[line_id] = 1
            line_id += 1
        return cur_index

    def write_data(cur_index, input_file, output_file):
        print(len(cur_index), input_file, output_file)
        out_file = open(output_file, 'w')
        line_id = 0
        for line in file(input_file):
            if line_id in cur_index:
                if '.x' in output_file:
                    out_file.write(line)
                elif '.y' in output_file:
                    out_file.write(line)
                else:
                    out_file.write(line)
            line_id += 1
        out_file.close()

    folder_test = join(folder_multi, str(train_id))
    folder_train = join(folder_multi, str(train_id), str(split_id))
    dict_test_id = load_obj(join(folder_test, 'split_' + str(train_id)))
    dict_id = load_obj(join(folder_train, 'split_' + str(split_id)))
    dict_date = {}
    dict_date['train'] = dict_id['train']
    dict_date['dev'] = dict_id['test']
    dict_date['test'] = dict_test_id['test']
    print len(dict_date['train']) + len(dict_date['test']) + len(dict_date['dev'])
    index = {}
    for train in ['train', 'test', 'dev']:
        for man in ['man', 'man_wit', 'wit']:
            index[(train, man)] = get_index(train, man)
    for man in ['man', 'man_wit', 'wit']:
        for train in ['train', 'test', 'dev']:
            print len(index[(train, man)])
    list_file = {}
    input_file = {}
    for man in ['man', 'man_wit', 'wit']:
        if man == 'wit':
            list_postfix = ['x']
        else:
            list_postfix = ['x', 'y']
        for postfix in list_postfix:
            list_file[(man, 'test', postfix)] = join(folder_test, man + '.test.' + postfix + '.txt')
            for prefix in ['train', 'dev']:
                list_file[(man, prefix, postfix)] = join(folder_train, man + '.' + prefix + '.' + postfix + '.txt')
            input_file[(man, postfix)] = join(folder_multi,  man + '.' + postfix + '.txt')
    for man in ['man', 'man_wit', 'wit']:
        list_file[(man, 'test', 'info')] = join(folder_test, man + '.test.' + 'info.txt')
        for prefix in ['train', 'dev']:
            list_file[(man, prefix, 'info')] = join(folder_train, man + '.' + prefix + '.' + 'info.txt')
        input_file[(man, 'info')] = join(folder_multi, man + '.info.txt')
    for man in ['man', 'man_wit', 'wit']:
        for prefix in ['train', 'test', 'dev']:
            if man == 'wit':
                list_postfix = ['x']
            else:
                list_postfix = ['x', 'y']
            for postfix in list_postfix:
                write_data(index[prefix, man], input_file[(man, postfix)], list_file[(man, prefix, postfix)])
            write_data(index[prefix, man], input_file[(man, 'info')], list_file[(man, prefix, 'info')])


def get_train_data(train_id, split_id, train):
    folder_train = join(folder_multi, str(train_id), str(split_id))
    folder_error = join(folder_train, 'train')    
    if not exists(folder_error):
        os.makedirs(folder_error)
    list_x = []
    list_y = []
    list_info = []
    for line in file(join(folder_train, 'man.' + train + '.x.txt')):
        list_x.append(line.strip('\n'))
    for line in file(join(folder_train, 'man_wit.' + train + '.x.txt')):
        list_x.append(line.strip('\n').split('\t')[0])
    for line in file(join(folder_train, 'man.' + train + '.y.txt')):
        list_y.append(line)
    for line in file(join(folder_train, 'man_wit.' + train + '.y.txt')):
        list_y.append(line)
    for line in file(join(folder_train, 'man.' + train + '.info.txt')):
        list_info.append(line)
    for line in file(join(folder_train, 'man_wit.' + train + '.info.txt')):
        list_info.append(line)
    index = np.arange(len(list_x))
    out_x = open(join(folder_error, train + '.x.txt'), 'w')
    out_y = open(join(folder_error, train + '.y.txt'), 'w')
    out_info = open(join(folder_error, train +  '.info.txt'), 'w')
    for i in index:
        out_x.write(list_x[i] + '\n')
        out_y.write(list_y[i])
        out_info.write(list_info[i])
    out_x.close()
    out_y.close()
    out_info.close()


folder_multi = '/gss_gpfs_scratch/dong.r/Dataset/OCR/book1800/single'
arg_num_fold = int(sys.argv[1])
arg_train_id = int(sys.argv[2])
split_train_test(arg_num_fold, arg_train_id)
arg_dev_id = int(sys.argv[3])
get_rand_train_dev(arg_dev_id)
split_train_dev(arg_num_fold, arg_train_id, arg_dev_id)
print ('Splitting Data')
split_date(arg_train_id, arg_dev_id)
print ('Get Training and Dev data')
get_train_data(arg_train_id, arg_dev_id, 'train')
get_train_data(arg_train_id, arg_dev_id, 'dev')
