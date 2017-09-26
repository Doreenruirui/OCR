import os
from os.path import join, exists
import numpy as np
import sys
from collections import OrderedDict

# folder_data = '/Users/doreen/Documents/Experiment/dataset/OCR/'
folder_data = '/scratch/dong.r/Dataset/OCR'


def evaluate_by_book(cur_folder, prefix, eval_prefix, col, nline):
    cur_folder = join(folder_data, cur_folder)
    info_file = join(cur_folder, prefix + '.info.txt')
    error_file = join(cur_folder, eval_prefix + '.ec.txt')
    books = []
    uni_books = OrderedDict()
    for line in file(info_file):
        cur_book = line.split('\t')[0]
        books.append(cur_book)
        if cur_book not in uni_books:
            uni_books[cur_book] = 1
    last_book = ''
    line_id = 0
    mean_e = []
    sum_e = 0
    sum_l = 0
    out_file = open(join(cur_folder, 'book.' + eval_prefix + '.ec.txt'), 'w')
    for line in file(error_file):
        cur_book = books[line_id]
        items = map(float, line.strip('\n').split())
        if cur_book != last_book:
            if line_id > 0:
                out_file.write(last_book + '\t' + str(len(mean_e)) + '\t' + str(np.mean(mean_e)) + '\t' + str(sum_e / sum_l) + '\n')
                sum_e = 0
                sum_l = 0
                mean_e = []
            last_book = cur_book
        sum_e += items[col]
        sum_l += items[-1]
        mean_e.append(items[col] / items[-1])
        line_id += 1
        if line_id == nline:
            break
    out_file.write(last_book + '\t' + str(len(mean_e)) + '\t' + str(np.mean(mean_e)) + '\t' + str(sum_e / sum_l) + '\n')
    out_file.close()


def evaluate_by_book_new(cur_folder, prefix, eval_prefix, col, nline):
    cur_folder = join(folder_data, cur_folder)
    info_file = join(cur_folder, prefix + '.info.txt')
    error_file = join(cur_folder, eval_prefix + '.ec.txt')
    books = []
    uni_books = OrderedDict()
    for line in file(info_file):
        cur_book = line.split('\t')[0]
        books.append(cur_book)
        if cur_book not in uni_books:
            uni_books[cur_book] = []
    last_book = ''
    line_id = 0
    for line in file(error_file):
        cur_book = books[line_id]
        items = map(float, line.strip('\n').split())
        uni_books[cur_book].append([items[col], items[-1]])
        line_id += 1
        if line_id == nline:
            break
    out_file = open(join(cur_folder, 'book.' + eval_prefix + '.ec.txt'), 'w')
    for book in uni_books:
        mean_e = sum([ele[0] / ele[1] for ele in uni_books[book]]) / len(uni_books[book])
        macro_e = sum([ele[0] for ele in uni_books[book]]) / sum([ele[1] for ele in uni_books[book]])
        out_file.write(book + '\t' + str(len(uni_books[book])) + '\t' + str(mean_e) + '\t' + str(macro_e) + '\n')
    out_file.close()

arg_folder = sys.argv[1]
arg_prefix = sys.argv[2]
arg_eval_prefix = sys.argv[3]
arg_col = int(sys.argv[4])
arg_nline = int(sys.argv[5])
evaluate_by_book_new(arg_folder, arg_prefix, arg_eval_prefix, arg_col, arg_nline)
