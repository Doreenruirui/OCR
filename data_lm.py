from PyLib.operate_file import save_obj, load_obj
from os.path import join, exists
import os
import sys
from util_lm import remove_nonascii


folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR/'


def write_file_train_all(cur_folder, error_ratio, train_id, split_id):
    global folder_data
    folder_data = join(folder_data, cur_folder)
    folder_train = join(folder_data, str(train_id), str(split_id))
    folder_out = join(folder_train, str(error_ratio))
    if not exists(folder_out):
        os.makedirs(folder_out)
    dict_id = load_obj(join(folder_train, 'split_' + str(split_id)))
    train_date = dict_id['train']
    file_list = os.listdir(join('/gss_gpfs_scratch/dong.r/Dataset/unprocessed/richmond', 'richmond-dispatch-correct'))
    f_ = open(join(folder_out, 'train.text'), 'w')
    num_train = 0
    for fn in file_list:
        if not fn.startswith('.'):
            cur_date = fn[:-2]
            if cur_date in train_date:
                num_train += 1
                for line in file(join('/gss_gpfs_scratch/dong.r/Dataset/unprocessed/richmond', 'richmond-dispatch-correct', fn)):
                     if len(line.strip()) > 0 and 'Column: ' not in line:
                         cur_str =  remove_nonascii(line[:-1].replace('###UNCLEAR###', '  '))
                         cur_str = ' '.join([ele for ele in cur_str.split() if len(ele.strip()) > 0])
                         f_.write(cur_str + '\n')
    f_.close()
    print num_train


write_file_train_all(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
