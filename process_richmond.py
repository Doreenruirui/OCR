from os.path import join, exists
import json
from PyLib.operate_file import save_obj, load_obj
from levenshtein import align, align_pair
from multiprocessing import Pool
import os
import re

folder_data = '/scratch/dong.r/Dataset/OCR/richmond'


def load_pairs():
    keys = ['b1', 'e1', 'b2', 'e2']
    info_file = open(join(folder_data, 'pair_info'), 'w')
    pair_file_x = open(join(folder_data, 'pair_x'), 'w')
    pair_file_y = open(join(folder_data, 'pair_y'), 'w')
    for line in file(join(folder_data, 'pairs-n5.json')):
        cur_dict = json.loads(line.strip())
        for pair in cur_dict['pairs']:
            str1 = re.sub('\n', ' ', pair['_1'].strip())
            str2 = re.sub('\n', ' ', pair['_2'].strip())
            pair_file_y.write(str1.encode('utf-8') + '\n')
            pair_file_x.write(str2.encode('utf-8') + '\n')
            cur_info = cur_dict['id1'][:12] + '\t' + '\t'.join(map(str, [cur_dict[ele] for ele in keys]))
            info_file.write(cur_info + '\n')
    info_file.close()
    pair_file_x.close()
    pair_file_y.close()



def split_dataset_with_date(train_ratio, error_ratio, split_id):
    file_list = os.listdir(join(folder_data, 'richmond-dispatch-correct'))
    date_list = list(set([ele[:-2] for ele in file_list if not ele.startswith('.')]))
    num_file = len(date_list)
    split_with_ratio()
    rand_index = np.arange(num_file)
    np.random.shuffle(rand_index)
    num_train = int(np.floor(num_file * train_ratio))
    num_test = num_file - num_train
    num_dev = num_train - int(np.floor(num_train * train_ratio))
    num_train = num_file - num_test - num_dev
    train_id = np.sort(rand_index[:num_train])
    dev_id = np.sort(rand_index[num_train: num_train + num_dev])
    test_id = np.sort(rand_index[-num_test:])
    train_date = OrderedDict()
    for index in train_id:
        train_date[date_list[index]] = 1
    test_date = OrderedDict()
    for index in test_id:
        test_date[date_list[index]] = 1
    dev_date = OrderedDict()
    for index in dev_id:
        dev_date[date_list[index]] = 1
    save_obj(join(folder_data, 'date_split_' + str(error_ratio) + '_' + str(split_id)),
             {'train_date': train_date, 'dev_date': dev_date, 'test_date': test_date})


def write_file(error_ratio, split_id):
    def write_file(prefix, data_id):
        with open(join(folder_out, prefix + '.x.txt'), 'w') as f_:
            for cur_id in data_id:
                f_.write(x_list[cur_id] + '\n')
        with open(join(folder_out, prefix + '.y.txt'), 'w') as f_:
            for cur_id in data_id:
                f_.write(y_list[cur_id] + '\n')
    dict_id = load_obj(join(folder_data, 'date_split_' + str(error_ratio) + '_' + str(split_id)))
    train_date = dict_id['train_date']
    test_date = dict_id['test_date']
    # dev_date = dict_id['dev_date']
    pairs_info = []
    for line in file(join(folder_data, 'pair_info')):
        date = line.split('\t')[0][:-2]
        pairs_info.append(date)
    pair_dis = np.loadtxt(join(folder_data, 'pairs_error'))
    x_list = []
    y_list = []
    for line in file(join(folder_data, 'pairs')):
        y, x = line.strip().split('\t\t')
        if len(x) > 0 and len(y) > 0:
            x_list.append(x)
            y_list.append(y)
    num_pair = len(pairs_info)
    train_id = OrderedDict()
    test_id = OrderedDict()
    dev_id = OrderedDict()
    for i in range(num_pair):
        if 'UNCLEAR' in y_list[i]:
            continue
        cur_date = pairs_info[i]
        if cur_date in train_date:
            if pair_dis[i] < 0.01 * error_ratio:
                train_id[i] = 1
        elif cur_date in test_date:
            test_id[i] = 1
        else:
            dev_id[i] = 1
    print len(train_id), len(dev_id), len(test_id)

    print len(x_list)
    print len(y_list)
    folder_out = join(folder_data, 'char_date_' + str(error_ratio) + '_' + str(split_id)+'_new')
    if not exists(folder_out):
        os.makedirs(folder_out)
    write_file('train', train_id)
    write_file('test', test_id)
    write_file('dev', dev_id)


def get_text_for_lm(error_ratio, split_id):
    dict_id = load_obj(join(folder_data, 'date_split_' + str(error_ratio) + '_' + str(split_id)))
    train_date = dict_id['train_date']
    file_list = os.listdir(join(folder_data, 'richmond-dispatch-correct'))
    folder_out = join(folder_data, 'char_date_' + str(error_ratio) + '_' + str(split_id))
    f_ = open(join(folder_out, 'text'), 'w')
    num_train = 0
    for fn in file_list:
        if not fn.startswith('.'):
            cur_date = fn[:-2]
            if cur_date in train_date:
                num_train += 1
                for line in file(join(folder_data, 'richmond-dispatch-correct', fn)):
                    if len(line.strip()) > 0 and 'Column: ' not in line:
                        f_.write(line.replace('###UNCLEAR### ', ''))
    f_.close()
    print num_train


load_pairs()
