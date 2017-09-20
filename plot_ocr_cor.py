from plot_curve import plot
from os.path import join
import numpy as np
import sys


# folder_data = '/scratch/dong.r/Dataset/OCR'
folder_data = '/home/rui/Dataset/OCR'


def plot_ocr_cor(cur_folder, file1, file2):
    cur_folder = join(folder_data, cur_folder)
    e1 = []
    for line in file(join(cur_folder, file1)):
        items = map(float, line.strip('\n').split('\t'))
        e1.append(items[0] / items[-1])
    dis2 = np.loadtxt(join(cur_folder, file2))
    e2 = dis2[:, 0] / dis2[:, -1]
    filename = 'OCR_vs_corrected_error_rate.png'
    title = 'OCR Error Rate v.s. Corrected Error Rate'
    xlabel = 'OCR Error Rate'
    ylabel = 'Corrected Error Rate'
    xlim = [0, 1]
    ylim = [0, 1]
    lenlabel = []
    fig = 1
    plot(e1, [e2], xlabel, ylabel, xlim, ylim, lenlabel, title, fig, filename)

plot_ocr_cor(sys.argv[1], sys.argv[2], sys.argv[3])