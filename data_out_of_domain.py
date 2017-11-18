import string
import os
import sys
import numpy as np
from multiprocessing import Pool

# trash_text = ['Distributed Proofreading Team at http://www.pgdp.net (This book was produced from scanned images of public domain material from the Google Print project.',
#               'Transcriber\'s Notes:',
#               '1. Passages in italics are surrounded by _underscores_. Passages in Decorative Fonts are surrounded by =equals=. Superscripted numbers are pre     ceded by a ^carat. Multiple superscripted numbers are surrounded by curly brackets {1 2}.',
#               '2. Corrections from the "Errata" page have been incorporated into this e-text.',
#               '3. Horizontal tables exceeding the width of this e-text have been reformatted to fit vertically.'
#               '4. Additional Transcriber\'s Notes are located at the end of this e-text.']


def one_book(bookname):
    with open(bookname, 'r') as f_:
        lines = f_.readlines()
    non_empty_lines = []
    for line in lines:
        old_line = line.strip('\n')
        line = line.strip('\n').translate(None, string.punctuation)
        line = line.replace('\t', ' ')
        line = ' '.join([ele for ele in line.split(' ') if len(ele) > 0])
        if len(line) > 1:
            non_empty_lines.append(old_line)
    output_lines = []
    for line in non_empty_lines:
        num_char = 0
        len_line = len(line)
        while num_char < len_line:
            cur_num = int(np.floor(np.random.randn() * 5 + 45))
            cur_num = min(70, cur_num)
            cur_num = max(0, cur_num)
            cur_num = min(cur_num, len_line - num_char)
            cur_str = line[num_char : num_char + cur_num]
            output_lines.append(cur_str)
            num_char += cur_num
    return output_lines


# def process_books(cur_folder):
#     list_file = [os.path.join(cur_folder, ele) for ele in os.listdir(cur_folder)]
#     # one_book(list_file[0])
#     pool = Pool(8)
#     pool.map(one_book, list_file)


def get_books(input_folder, output_folder, max_lines):
    input_files = [ele for ele in os.listdir(input_folder)
                   if ele.endswith('.txt') and not ele.startswith('.')]
    num_file = len(input_files)
    rand_index = np.arange(num_file)
    np.random.shuffle(rand_index)
    num_line = 0
    max_train = int(np.floor(max_lines * 0.8))
    file_no = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    f_y = open(os.path.join(output_folder, 'train.y.txt'), 'w')
    f_info = open(os.path.join(output_folder, 'train.info.txt'), 'w')
    while num_line < max_train:
        print num_line
        cur_no = rand_index[file_no]
        cur_file = input_files[cur_no]
        lines = one_book(os.path.join(input_folder, cur_file))
        cur_num = len(lines)
        true_num = min(cur_num, max_train - num_line)
        f_y.write('\n'.join(lines[:true_num]) + '\n')
        f_info.write((input_files[cur_no].split('.txt')[0] + '\n') * true_num)
        num_line += true_num
        file_no += 1
    f_y.close()
    f_info.close()
    f_y = open(os.path.join(output_folder, 'dev.y.txt'), 'w')
    f_info = open(os.path.join(output_folder, 'dev.info.txt'), 'w')
    while num_line < max_lines:
        print num_line
        cur_no = rand_index[file_no]
        cur_file = input_files[cur_no]
        lines = one_book(os.path.join(input_folder, cur_file))
        cur_num = len(lines)
        true_num = min(cur_num, max_lines - num_line)
        f_y.write('\n'.join(lines[:true_num]) + '\n')
        f_info.write((input_files[cur_no].split('.txt')[0] + '\n') * true_num)
        num_line += true_num
        file_no += 1
    f_y.close()
    f_info.close()


get_books(sys.argv[1], sys.argv[2], int(sys.argv[3]))









# process_books(sys.argv[1])
