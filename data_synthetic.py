from os.path import join
import os


folder_multi = '/scratch/dong.r/Dataset/OCR/multi'


def get_train_data(train_id, split_id, error_ratio, train):
    folder_train = join(folder_multi, str(train_id), str(split_id))
    folder_out = join(folder_train, str(error_ratio) + '_syn')
    list_y = []
    for line in file(join(folder_train, train + '.y.txt')):
        list_y.append(line)
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    f_x = open(join(folder_out, train + '.x.txt'), 'w')
    f_y = open(join(folder_out, train + '.y.txt'), 'w')
    for i in range(len(list_y)):
        cur

