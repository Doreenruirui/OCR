import os
from os.path import join, exists
import numpy as np
import sys
from collections import OrderedDict
from plot_curve import plotBar, plot

# folder_data = '/Users/doreen/Documents/Experiment/dataset/OCR/'
folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR'
#folder_data = '/home/rui/Dataset/OCR'


def merge_error_rate(cur_folder):
    cur_folder = join(folder_data, cur_folder)
    error = [[], [], []]
    num_line = []
    books = []
    for line in file(join(cur_folder, 'book.man_wit.test.ec.txt')):
        items = line.strip('\n').split('\t')
        error[0].append(float(items[2]))
        num_line.append(int(items[1]))
        books.append(items[0])
    for line in file(join(cur_folder, 'book.man_wit.test.single.ec.txt')):
        items = line.strip('\n').split('\t')
        error[1].append(float(items[2]))
    for line in file(join(cur_folder, 'book.man_wit.test.avg.ec.txt')):
        items = line.strip('\n').split('\t')
        error[2].append(float(items[2]))
    ngroup = len(books)
    print 'AVG better than SINGLE:'
    num_avg = 0
    with open(join(cur_folder, 'error_rate_per_book.txt'), 'w') as f_:
        for i in range(ngroup):
            if error[2][i] < error[1][i]:
                num_avg += 1
                f_.write('\t'.join(map(str, [books[i], num_line[i], error[0][i], error[1][i], error[2][i]])) + '\n')
                # print books[i], num_line[i], error[0][i], error[1][i], error[2][i]
        # f_.write('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print num_avg
        print 'AVG worse than SINGLE:'
        num_avg = 0
        for i in range(ngroup):
            if error[2][i] > error[1][i]:
                num_avg += 1
                f_.write('\t'.join(map(str, [books[i], num_line[i], error[0][i], error[1][i], error[2][i]])) + '\n')
        print num_avg
    # stickers = books
    # stickers = [ele for ele in range(len(books))]
    # lenlabels = ['OCR', 'SINGLE', 'AVG']
    # xlabel = 'Book Name'
    # ylabel = 'Error Rate'
    # title = 'Error Rate Per Book'
    # figure_name = 'Results/Error_Rate_Per_Book.png'
    # error = [error[0][:10], error[1][:10], error[2][:10]]
    # plotBar(ngroup, error, stickers, lenlabels, xlabel, ylabel, title, figure_name, 0.2)
    # plot(stickers, error, xlabel, ylabel, [0, 380], [0, 1], lenlabels, title, figure_name)

arg_folder = sys.argv[1]
merge_error_rate(arg_folder)
