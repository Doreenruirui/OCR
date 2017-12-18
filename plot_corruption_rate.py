from plot_curve import plot
from os.path import join
import numpy as np
import sys
import matplotlib.pyplot as plt

# folder_data = '/scratch/dong.r/Dataset/OCR'
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
    dict_ER = {'CER-S': [0.1589885469, 0.160107996, 0.1617929973],
               'CER-M': [0.1248197813, 0.1255424109, 0.1269869924],
               'LCER-S': [0.1172924516, 0.1170423416, 0.1168972954],
               'LCER-M': [0.08663468203, 0.08662940498, 0.08669897879],
               'WER-S': [0.3472413896, 0.3468747795, 0.3465946598],
               'WER-M': [0.2744800659, 0.2751123637, 0.2758492336],
               'LWER-S': [0.2561318143, 0.2530116959, 0.249482516],
               'LWER-M': [0.1874299311, 0.1859763062, 0.1843808998]}
    dict_lim = {'CER': [0.12, 0.18], 'WER': [0.27, 0.38], 'LCER': [0.08, 0.13], 'LWER': [0.15, 0.3]}
    i = 0
    y = []
    for decode in ['-S', '-M']:
        y.append(dict_ER[name_error + decode])
    plot_error(name_error, y, dict_lim[name_error])
    i += 1


def plot_error(name_error, y, ylim):
    x = [0.09, 0.12, 0.15]
    filename = 'Results/Corrupt_Rate_%s.png' % name_error
    ylabel = '%s (macro-avg)' % name_error
    xlabel = 'Corruption Rate'
    xlim = [0.089, 0.151]
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
    plt.tight_layout()
    plt.savefig(filename)


#
# def plot_error(name_error, y, ylim, ngroup):
#     x = (np.arange(ngroup) + 3) * 0.3
#     xstickers = [str(ele) for ele in x]
#     print x, xstickers
#     lenlabel = ['Single-Input', 'Multi-Input']
#     filename = 'Results/Error Rate V.S. Corruption Rate_%s.png' % name_error
#     ylabel = 'Error Rate'
#     xlabel = 'Corruption Rate'
#     xlim = [0.7, 1.6]
#     index = x
#     bar_width = 0.04
#     opacity = 0.4
#     error_config = {'ecolor': '0.3'}
#     colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c']
#     rects = []
#     for i in range(len(y)):
#         rects.append(plt.bar(index - bar_width * 2 + bar_width * i, y[i], bar_width,
#                              alpha=opacity,
#                              color=colors[i],
#                              error_kw=error_config,
#                              label=lenlabel[i]))
#
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.xlim(xlim)
#     plt.ylim(ylim)
#     plt.xticks(fontsize=8)
#     plt.yticks(fontsize=8)
#     # plt.title(title)
#     plt.xticks(index - bar_width * 1, xstickers)
#     plt.legend(borderpad=1, bbox_transform=plt.gcf().transFigure, loc=1, fontsize=10)
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(filename)

error='CER'
main(error)

