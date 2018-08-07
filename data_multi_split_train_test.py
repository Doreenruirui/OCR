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


def get_wit_and_man_date():
    dict_date = {}
    for line in file(join(folder_multi, 'man.info.txt')):
        items = line.split('\t')
        cur_date = items[0]
        if cur_date not in dict_date:
            dict_date[cur_date] = 1
    return dict_date


def check_manual():
    dict_date = get_wit_and_man_date()
    for line in file(join(folder_multi, 'man_wit.info.txt')):
        items = line.split('\t')
        cur_date = items[0]
        if cur_date in dict_date:
            dict_date[cur_date] = 2
    print len([ele for ele in dict_date if dict_date[ele] == 2])
    # return dict_date


def get_all_date():
    dict_date = OrderedDict()
    for line in file(join(folder_multi, 'man.info.txt')):
        items = line.split('\t')
        cur_date = items[0]
        if cur_date not in dict_date:
            dict_date[cur_date] = 1
    save_obj(join(folder_multi, 'date_info'), dict_date)



def split_train_test(train_ratio, split_id):
    dict_date = load_obj(join(folder_multi, 'date_info'))
    folder_split = join(folder_multi, str(split_id))
    if not exists(folder_split):
        os.makedirs(folder_split)
    date_list = dict_date.keys()
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
    folder_train = join(folder_multi, str(train_id))
    folder_split = join(folder_multi, str(train_id), str(split_id))
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



def split_date(train_id, split_id):
    def get_index(name_train, name_man):
        index = {}
        line_id = 0
        for line in file(join(folder_multi, name_man + '.info.txt')):
            items = line.split('\t')
            cur_date = items[0]
            if name_train == 'train':
                if cur_date not in dict_date['test'] and cur_date not in dict_date['dev']:
                    index[line_id] = 1
            else:
                if cur_date in dict_date[name_train]:
                    index[line_id] = 1
            line_id += 1
        return index
    def write_data(index, input_file, output_file):
        print(len(index), input_file, output_file)
        out_file = open(output_file, 'w')
        line_id = 0
        for line in file(input_file):
            if line_id in index:
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
        #for man in ['wit']:
        for man in ['man', 'man_wit']:
            index[(train, man)] = get_index(train, man)
    for man in ['man', 'man_wit']:
    #for man in ['wit']:
        for train in ['train', 'test', 'dev']:
            print len(index[(train, man)])
    list_file = {}
    input_file = {}
    for man in ['man', 'man_wit']:
    #for man in ['wit']:
        #for postfix in ['x']:
        for postfix in ['x', 'y']:
            list_file[(man, 'test', postfix)] = join(folder_test, man + '.test.' + postfix + '.txt')
            for prefix in ['train', 'dev']:
                list_file[(man, prefix, postfix)] = join(folder_train, man + '.' + prefix + '.' + postfix + '.txt')
            input_file[(man, postfix)] = join(folder_multi,  man +'.' + postfix + '.txt')
    for man in ['man', 'man_wit']:
    #for man in ['wit']:
        list_file[(man, 'test', 'info')] = join(folder_test, man + '.test.' + 'info.txt')
        for prefix in ['train', 'dev']:
            list_file[(man, prefix, 'info')] = join(folder_train, man + '.' + prefix + '.' + 'info.txt')
        input_file[(man, 'info')] = join(folder_multi, man + '.info.txt')
    for man in ['man', 'man_wit']:
    #for man in ['wit']:
        for prefix in ['train', 'test', 'dev']:
            #for postfix in ['x']:
            for postfix in ['x', 'y']:
                write_data(index[prefix, man], input_file[(man, postfix)], list_file[(man, prefix, postfix)])
            write_data(index[prefix, man], input_file[(man, 'info')], list_file[(man, prefix, 'info')])


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


def get_train_data(train_id, split_id, error_ratio, train):
    folder_train = join(folder_multi, str(train_id), str(split_id))
    #folder_error = join(folder_train, str(error_ratio))
    folder_error = join(folder_train, 'all')                      
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
    #dis = np.loadtxt(join(folder_train, 'distance'))
    #if train == 'train':
    #    index = []
    #    for i in range(len(list_x)):
    #        if dis[i] * 1. / len(list_y[i]) <= error_ratio * 0.01:
    #            index.append(i)
    #else:
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

folder_multi = '/gss_gpfs_scratch/dong.r/Dataset/OCR/richmond/single'
# check_manual()
get_all_date()
#split_train_test(0.8, 0)
#split_train_dev(0.8, 0, 0)
cur_test_id =  int(sys.argv[1])
cur_train_id = int(sys.argv[2])
cur_error = int(sys.argv[3])
#split_train_test(0.8, cur_test_id)
#split_train_dev(0.8, cur_test_id, cur_train_id)
# print ('Splitting Data')
split_date(cur_test_id, cur_train_id)
# print ('Computing Error Rate')
#compute_error_rate(cur_test_id, cur_train_id)
print ('Get Training and Dev data')
get_train_data(cur_test_id, cur_train_id, cur_error, 'train')
get_train_data(cur_test_id, cur_train_id, cur_error, 'dev')
