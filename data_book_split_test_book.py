from os.path import exists, join
from collections import OrderedDict
import os
import sys


folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR'


def split_test_book(cur_folder, prefix):
    cur_folder = join(folder_data, cur_folder)
    info_file = join(cur_folder, prefix + '.info.txt')
    out_folder = join(cur_folder, 'books')
    if not exists(out_folder):
        os.makedirs(out_folder)
    books = []
    for line in file(info_file):
        books.append(line.split('\t')[0])
    f_all = open(join(cur_folder, 'test_books'), 'w')
    for ele in set(books):
        f_all.write(ele + '\n')
    f_all.close()
    if 'man' in prefix:
        list_postfix = ['x', 'y']
    else:
        list_postfix = ['x']
    for postfix in list_postfix:
        last_book = ''
        line_id = 0
        f_ = None
        cur_file = join(cur_folder, prefix + '.' + postfix + '.txt')
        for line in file(cur_file):
            cur_book = books[line_id]
            if cur_book != last_book:
                if line_id > 0:
                    f_.close()
                out_file = join(out_folder, cur_book + '.' + prefix + '.' + postfix + '.txt')
                f_ = open(out_file, 'w')
                f_.write(line)
                last_book = cur_book
            else:
                f_.write(line)
            line_id += 1
        f_.close()


arg_folder = sys.argv[1]
split_test_book(arg_folder, 'man_wit.test')
split_test_book(arg_folder, 'man.test')
# split_test_book(arg_folder, 'wit.test')


