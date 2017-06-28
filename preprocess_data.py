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
            for pair in cur_dict['pairs'][1:-1]:
                str1 = pair['_1'].strip().replace('\n', ' ')
                str2 = pair['_2'].strip().replace('\n', ' ')
                if len(str1) > 0 and len(str2) > 0:
                    if str2[-1] == '-' and str1[-2] != '-':
                        str2 = str2[:-1]
                    # if '###UNCLEAR###' in str1:
                    #     dis = align(str1.replace('###UNCLEAR###', ''), str2)
                    # else:
                    #     dis = align(str1, str2)
                    pair_file.write(str1.encode('utf-8') + '\t\t' + str2.encode('utf-8') + '\n')
                    cur_info = cur_dict['id1'][:12] + '\t' + '\t'.join(map(str, [cur_dict[ele] for ele in keys]))
                    info_file.write(cur_info + '\n')
                    list_str1.append(str1.encode('utf-8'))
                    list_str2.append(str2.encode('utf-8'))
                    #
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


def load_pairs(error_ratio):
    keys = ['b1', 'e1', 'b2', 'e2']
    info_file = open(join(folder_data, 'pair_info_' + str(error_ratio)), 'w')
    pair_file = open(join(folder_data, 'pairs_' + str(error_ratio)), 'w')
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
                    if str1[-1] == '-' and str2[-1] != '-':
                        str1 = str1[:-1]
                    dis = align(str1, str2)
                    if dis < error_ratio * 0.01 * len(str2):
                        # print dis
                        pair_file.write(str1.encode('utf-8') + '\t\t' + str2.encode('utf-8') + '\n')
                        cur_info = cur_dict['id1'][:12] + '\t' + \
                                   '\t'.join(map(str, [cur_dict[ele] for ele in keys]))
                        info_file.write(cur_info + '\n')
    info_file.close()
    pair_file.close()
    print (num_pair)


def split_dataset(train_ratio, error_ratio):
    if not exists(join(folder_data, 'char_' + str(error_ratio))):
        os.makedirs(join(folder_data, 'char_' + str(error_ratio)))

    def write_file(prefix, data_id):
        with open(join(folder_data, 'char_' + str(error_ratio), prefix + '.x.txt'), 'w') as f_:
            for cur_id in data_id:
                f_.write(x_list[cur_id] + '\n')
        with open(join(folder_data, 'char_' + str(error_ratio), prefix + '.y.txt'), 'w') as f_:
            for cur_id in data_id:
                f_.write(y_list[cur_id] + '\n')
    with open(join(folder_data, 'pairs_' + str(error_ratio)), 'r') as f_:
        num_data = len(f_.readlines())
    #num_train = int(np.floor(num_data * train_ratio))
    #num_test = num_data - num_train
    #num_dev = num_train - int(np.floor(num_train * train_ratio))
    #num_train = num_data - num_test - num_dev
    #rand_index = np.arange(num_data)
    #np.random.shuffle(rand_index)
    #train_id = np.sort(rand_index[:num_train])
    #dev_id = np.sort(rand_index[num_train: num_train + num_dev])
    #test_id = np.sort(rand_index[-num_test:])
    #save_obj(join(folder_data, 'pair_split_' + str(error_ratio)'),  {'train': train_id,
    #                                            'dev': dev_id,
    #                                            'test': test_id})
    dict_id = load_obj(join(folder_data, 'pair_split_' + str(error_ratio)))
    dev_id = dict_id['dev']
    train_id = dict_id['train']
    test_id = dict_id['test']
    print len(dict_id['dev'])
    print len(dict_id['train'])
    print len(dict_id['test'])
    x_list = []
    y_list = []
    for line in file(join(folder_data, 'pairs_' + str(error_ratio))):
        y, x = line.strip().split('\t\t')
        if len(x) > 0 and len(y) > 0:
            x_list.append(x)
            y_list.append(y)
    print len(x_list)
    print len(y_list)
    #print x_list[dict_id['train'][0]]
    #print x_list[dict_id['train'][-1]]
    #print x_list[dict_id['test'][0]]
    #print x_list[dict_id['test'][-1]]
    #write_file('train', train_id)
    #write_file('test', test_id)
    write_file('dev', dev_id)


load_pairs_all()
# load_pairs(error_ratio=25)
# split_dataset(train_ratio=0.8, error_ratio=25)
