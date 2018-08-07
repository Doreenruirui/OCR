from plot_curve import plot
from os.path import join
import numpy as np
import sys
import matplotlib.pyplot as plt

# folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR'
folder_data = '/home/rui/Dataset/OCR'


def main(name_error):
    # dict_ER = {'CER-S': [0.1007269538, 0.09594425986, 0.09353844304, 0.09170161632, 0.08986721349],
    #             'CER-M': [0.08499500909, 0.081568109, 0.07989229659, 0.07863544576, 0.07682741443],
    #             'LCER-S': [0.05423742351, 0.04960213105, 0.0475474423, 0.04591917635, 0.04487802975],
    #             'LCER-M': [0.04354314197, 0.04010959096, 0.03867574922, 0.03748838366, 0.03648857501],
    #             'WER-S': [0.1942108895, 0.1926249861, 0.1867630233, 0.1827663173, 0.1795267732],
    #             'WER-M': [0.1749140193, 0.1652614381, 0.1610879705, 0.1577572434, 0.1546135483],
    #             'LWER-S': [0.1053005654, 0.1028587273, 0.09800792104, 0.09444794382, 0.09253052077],
    #             'LWER-M': [0.09059716655, 0.08201245883, 0.07852629404, 0.07588731916, 0.07403860613]}
    dict_ER = {'CER-S': [0.05245, 0.04958, 0.04835, 0.04837, 0.04843],
               'CER-M': [0.04742,0.04518,0.04449,0.04514,0.04505],
               'LCER-S': [0.01751,0.01606,0.01528,0.01489,0.01481],
               'LCER-M': [0.01352,0.01255,0.01225,0.01221,0.01208],
               'WER-S': [0.13117,0.12336,0.11994,0.11941,0.11941],
               'WER-M': [0.11732,0.11133,0.10854,0.10917,0.10904],
               'LWER-S': [0.04291,0.03822,0.03574,0.03451,0.03435],
               'LWER-M': [0.03072,0.02760,0.02621,0.02579,0.02558]}
    dict_lim = {'CER': [0.04, 0.06], 'WER': [0.10, 0.14], 'LCER': [0.01, 0.018], 'LWER': [0.02, 0.05]}
    i = 0
    y = []
    for decode in ['-S', '-M']:
        y.append(dict_ER[name_error + decode])
    plot_error_length(name_error, dict_lim[name_error], y)
    i += 1


def plot_error_length(name_error, ylim, y):
    x = [0.2, 0.4, 0.6, 0.8, 1]
    filename = 'Results/Book_Training_Size_%s.png' % name_error
    ylabel = '%s (macro-avg)' % name_error
    xlabel = 'Training Percent'
    xlim = [0.199, 1.001]
    lenlabel = ['Single Input', 'Multiple Input']
    plt.figure(figsize=(8, 10))
    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(x, [str(ele) for ele in x], fontsize=25)
    plt.yticks(fontsize=25)
    colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c']
    cs = [ele + '-' for ele in colors]
    dot = ['o', '^', '*', 'd', 's', '*']
    if len(y) > 1:
        for i in range(len(y)):
            plt.plot(x, y[i], cs[i], label=lenlabel[i], linewidth=3.3)
            plt.scatter(x, y[i], c='c', s=15, marker=dot[i])
        # plt.legend(bbox_to_anchor=(0, 0.26, 0.96, 1), bbox_transform=plt.gcf().transFigure, loc=4, fontsize=6.5)
        plt.legend(borderpad=1, bbox_transform=plt.gcf().transFigure, loc=1, fontsize=25)
        # leg = plt.gca().get_legend()
        # leg.draw_frame(False)
        # ltext = leg.get_texts()  # all the text.Text instance in the legend
        # llines = leg.get_lines()  # all the lines.Line2D instance in the legend
        # plt.setp(ltext, fontsize=6)  # the legend text fontsize
        # plt.setp(llines, linewidth=2)  # the legend linewidth
    else:
        plt.plot(x, y[0], '-o')
    # plt.axes().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(filename)

error='LWER'
main(error)

