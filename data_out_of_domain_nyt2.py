import string
import os
import sys
import numpy as np
from os.path import join as pjoin
from multiprocessing import Pool


def one_book(paras):
    bookname, infolder, outfolder = paras
    with open(pjoin(infolder, bookname), 'r') as f_:
        lines = f_.readlines()
    non_empty_lines = []
    for line in lines:
        old_line = line.strip('\n')
        line = line.strip('\n').translate(None, string.punctuation)
        line = line.replace('\t', ' ')
        line = ' '.join([ele for ele in line.split(' ') if len(ele) > 0])
        if len(line) > 1:
            non_empty_lines.append(old_line)
    # output_lines = []
    with open(pjoin(outfolder, bookname), 'w') as f_:
        for line in non_empty_lines:
            num_char = 0
            len_line = len(line)
            while num_char < len_line:
                cur_num = int(np.floor(np.random.randn() * 5 + 45))
                cur_num = min(70, cur_num)
                cur_num = max(0, cur_num)
                cur_num = min(cur_num, len_line - num_char)
                cur_str = line[num_char : num_char + cur_num]
                # output_lines.append(cur_str)
                f_.write(cur_str + '\n')
                num_char += cur_num
    # return output_lines


def get_all_data(input_folder, output_folder):
    input_files = [ele for ele in os.listdir(input_folder)
                   if ele.endswith('.txt') and ele.startswith('nyt_eng')]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    paras = zip(input_files, [input_folder for _ in input_files], [output_folder for _ in input_files])
    # one_book(paras[0])
    pool = Pool(100)
    pool.map(one_book, paras)

def merge_data(input_folder, output_folder):
    input_files = [ele for ele in os.listdir(input_folder)
                   if ele.endswith('.txt') and ele.startswith('nyt_eng')]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    f_info = open(pjoin(output_folder, 'train.info.txt'), 'w')
    with open(pjoin(output_folder, 'train.y.txt'), 'w') as f_:
        for fn in input_files:
            cur_file = pjoin(input_folder, fn)
            for line in file(cur_file):
                f_.write(line)
                f_info.write(fn[8:-4] + '\n')


# get_all_data(sys.argv[1], sys.argv[2])
merge_data(sys.argv[1], sys.argv[2])
#
#
#
#
# def get_books(input_folder, output_folder, max_lines):
#     input_files = [ele for ele in os.listdir(input_folder)
#                    if ele.endswith('.txt') and ele.startswith('nyt_eng')]
#     num_file = len(input_files)
#     rand_index = np.arange(num_file)
#     np.random.shuffle(rand_index)
#     num_line = 0
#     max_train = int(np.floor(max_lines * 0.8))
#     file_no = 0
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     f_y = open(os.path.join(output_folder, 'train.y.txt'), 'w')
#     f_info = open(os.path.join(output_folder, 'train.info.txt'), 'w')
#     while num_line < max_train:
#         print num_line
#         cur_no = rand_index[file_no]
#         cur_file = input_files[cur_no]
#         lines = one_book(os.path.join(input_folder, cur_file))
#         cur_num = len(lines)
#         true_num = min(cur_num, max_train - num_line, 10000)
#         cur_rand = np.arange(true_num)
#         np.random.shuffle(cur_rand)
#         f_y.write('\n'.join([lines[line_id] for line_id in cur_rand]) + '\n')
#         f_info.write((input_files[cur_no].split('.txt')[0] + '\n') * true_num)
#         num_line += true_num
#         file_no += 1
#     f_y.close()
#     f_info.close()
#     f_y = open(os.path.join(output_folder, 'dev.y.txt'), 'w')
#     f_info = open(os.path.join(output_folder, 'dev.info.txt'), 'w')
#     while num_line < max_lines:
#         print num_line
#         cur_no = rand_index[file_no]
#         cur_file = input_files[cur_no]
#         lines = one_book(os.path.join(input_folder, cur_file))
#         cur_num = len(lines)
#         true_num = min(cur_num, max_lines - num_line, 10000)
#         cur_rand = np.arange(true_num)
#         np.random.shuffle(cur_rand)
#         f_y.write('\n'.join([lines[line_id] for line_id in cur_rand]) + '\n')
#         f_info.write((input_files[cur_no].split('.txt')[0] + '\n') * true_num)
#         num_line += true_num
#         file_no += 1
#     f_y.close()
#     f_info.close()
#
#
# get_books(sys.argv[1], sys.argv[2], int(sys.argv[3]))



