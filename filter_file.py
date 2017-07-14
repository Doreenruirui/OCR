from my_utils import error_rate_line
import sys
import numpy as np
from os.path import exists, join
import os


def filter_file(infolder, prefix, error_ratio):
    id_x = []
    id_y = []
    line_id = 0
    infile = join(infolder, prefix)
    for line in file(infile + '.x.txt'):
        if len(line.strip()) == 0:
            id_x.append(line_id)
        line_id += 1
    line_id = 0
    for line in file(infile + '.y.txt'):
        if len(line.strip()) == 0:
            id_y.append(line_id)
        line_id += 1
    if error_ratio < 100:
        error_file = infile + '.error'
        if not exists(error_file):
            error_rate_line(error_file,
                            infile + '.x.txt',
                            infile + '.y.txt',
                            0,
                            -1,
                            flag_strip=1)
        pair_error = np.loadtxt(error_file, dtype=int)
        id_error = []
        for cur_id in range(pair_error.shape[0]):
            if pair_error[cur_id][2] > 0:
                cur_error = pair_error[cur_id][0] * 1. / pair_error[cur_id][2]
                if cur_error * 100 > error_ratio:
                    id_error.append(cur_id)
    else:
        id_error = []
    list_id = list(set(id_x) | set(id_y) | set(id_error))
    folder_out = join(infolder, str(error_ratio))
    if not exists(folder_out):
        os.makedirs(folder_out)
    new_x = open(join(folder_out, prefix + '.x.txt'), 'w')
    line_id = 0
    for line in file(infile + '.x.txt'):
        if line_id not in list_id:
            new_x.write(line.strip() + '\n')
        line_id += 1
    new_x.close()
    new_y = open(join(folder_out, prefix + '.y.txt'), 'w')
    line_id = 0
    for line in file(infile + '.y.txt'):
        if line_id not in list_id:
            new_y.write(line.strip() + '\n')
        line_id += 1
    new_y.close()

filter_file(sys.argv[1], sys.argv[2], int(sys.argv[3]))