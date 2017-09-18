from os.path import join, exists
import os
import sys
import re

folder_data = '/scratch/dong.r/Dataset/OCR'
arg_folder = sys.argv[1]
arg_train = sys.argv[2]
arg_out = sys.argv[3]

folder_train = join(folder_data, arg_folder)
folder_out = join(folder_data, arg_out)

list_x = []
index_y = {}

if not exists(folder_out):
    os.makedirs(folder_out)

with open(join(folder_out, arg_train + '.z.txt'), 'w') as f_:
    line_id = 0
    for line in file(join(folder_train, arg_train + '.z.txt')):
        line = re.sub('\xc5\xbf', 's', line)
        items = line.strip('\n').split('\t')
        if len(items) > 1:
            flag_filter = 0
            for ele in items[1:]:
                if ele != items[0]:
                    flag_filter = 1
                    break
            if flag_filter == 0:
                index_y[line_id] = 1
                f_.write(items[0] + '\n')
        else:
            index_y[line_id] = 1
            f_.write(line)
        line_id += 1


with open(join(folder_out, arg_train + '.x.txt'), 'w') as f_:
    line_id = 0
    for line in file(join(folder_train, arg_train + '.x.txt')):
        if line_id in index_y:
            f_.write(line)
        line_id += 1

with open(join(folder_out, arg_train + '.y.txt'), 'w') as f_:
    line_id = 0
    for line in file(join(folder_train, arg_train + '.y.txt')):
        if line_id in index_y:
            f_.write(line)
        line_id += 1
