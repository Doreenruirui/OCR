from plot_curve import plot
from os.path import join
import numpy as np
import sys
import matplotlib.pyplot as plt

# folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR'
folder_data = '/home/rui/Dataset/OCR'



# def plot_error_length():
#     origin = [0.1601347994, 0.1736580919, 0.1403431438, 0.0801415924, 0.07864798431, 0.08590460722, 0.09704840973, 0.1096014742, 0.1151836106, 0.1478754955, 0.2108129776, 0.2036823508]
#     single_top = [0.08208638443, 0.07599822117, 0.06619279, 0.03602172078, 0.02896132318, 0.03195126176, 0.0404151959, 0.05557161203, 0.05605953165, 0.08081657938, 0.1343615479, 0.1668379614]
#     single_best = [0.0219645146, 0.02843974216, 0.03509443075, 0.02112718414, 0.01682436225, 0.01832057505, 0.02392270729, 0.03540155231, 0.0347040667, 0.05112821012, 0.0933020662, 0.117605216]
#     avg_top = [0.09173952479, 0.08622220185, 0.06911295136, 0.0421775941, 0.03301228615, 0.03768383826, 0.03975876971, 0.05234117878, 0.0545027723, 0.07852274644, 0.137715882, 0.1737229261]
#     avg_best = [0.02095204209, 0.02906656511, 0.03354302665, 0.02219788657, 0.01804903104, 0.02073491292, 0.02230038804, 0.03023052683, 0.03160435876, 0.04600424454, 0.08331703392, 0.1258955211]
#     x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
#     y = []
#     y.append(origin)
#     y.append(single_top)
#     y.append(single_best)
#     y.append(avg_top)
#     y.append(avg_best)
#     filename = 'Results/Error Rate V.S. Input Length.png'
#     title = 'OCR Error Rate v.s. Input Length'
#     xlabel = 'OCR Error Rate'
#     ylabel = 'Input Length'
#     xlim = [0, 120]
#     ylim = [0, 0.5]
#     lenlabel = ['OCR', 'Single-Top', 'Single-Best', 'Avg-Top', 'Avg-Best']
#     fig = 1
#     plot(x, y, xlabel, ylabel, xlim, ylim, lenlabel, title, fig, filename)


def plot_error_length():
    origin = [0.1601347994, 0.1736580919, 0.1403431438, 0.0801415924, 0.07864798431, 0.08590460722, 0.09704840973, 0.1096014742, 0.1151836106, 0.1478754955, 0.2108129776, 0.2036823508]
    single_top = [0.08208638443, 0.07599822117, 0.06619279, 0.03602172078, 0.02896132318, 0.03195126176, 0.0404151959, 0.05557161203, 0.05605953165, 0.08081657938, 0.1343615479, 0.1668379614]
    single_best = [0.0219645146, 0.02843974216, 0.03509443075, 0.02112718414, 0.01682436225, 0.01832057505, 0.02392270729, 0.03540155231, 0.0347040667, 0.05112821012, 0.0933020662, 0.117605216]
    avg_top = [0.09173952479, 0.08622220185, 0.06911295136, 0.0421775941, 0.03301228615, 0.03768383826, 0.03975876971, 0.05234117878, 0.0545027723, 0.07852274644, 0.137715882, 0.1737229261]
    avg_best = [0.02095204209, 0.02906656511, 0.03354302665, 0.02219788657, 0.01804903104, 0.02073491292, 0.02230038804, 0.03023052683, 0.03160435876, 0.04600424454, 0.08331703392, 0.1258955211]
    x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    y = []
    y.append(np.asarray(origin))
    y.append(np.asarray(single_top))
    y.append(np.asarray(single_best))
    y.append(np.asarray(avg_top))
    y.append(np.asarray(avg_best))
    filename = 'Results/Error Rate V.S. Input Length_Book.png'
    title = 'OCR Error Rate v.s. Input Length for Book'
    ylabel = 'OCR Error Rate'
    xlabel = 'Input Length'
    xlim = [0, 120]
    ylim = [0, 0.3]
    lenlabel = ['OCR', 'Single-Top', 'Single-Best', 'Avg-Top', 'Avg-Best']
    # fig, ax = plt.subplots()
    index = np.asarray(x)
    bar_width = 1.2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c']
    rects = []
    for i in range(len(y)):
        rects.append(plt.bar(index - 6 + bar_width * i, y[i], bar_width,
                         alpha=opacity,
                         color=colors[i],
                         error_kw=error_config,
                         label=lenlabel[i]))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title)
    plt.xticks(index - bar_width * 2.5, [str(ele - 10) + '-' + str(ele) for ele in x])
    plt.legend(borderpad=2, bbox_transform=plt.gcf().transFigure, loc=2, fontsize=10)
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)


def plot_error_length_rich():
    origin = [0.287091746, 0.2907521349, 0.2480027316, 0.1442750806, 0.1292525302, 0.1917888109, 0.5421206403]
    single_top = [0.1559825816, 0.1688454502, 0.1488567924, 0.08094709119, 0.05545069724, 0.07989189764, 0.2528658312]
    single_best = [0.04850458402, 0.07387942036, 0.07649360759, 0.04353986608, 0.02412965871, 0.03795809342, 0.1609226547]
    avg_top = [0.1156612037, 0.1357071822, 0.1137644177, 0.05966595285, 0.02990594052, 0.02882328365, 0.06868507534]
    avg_best = [0.02362264276, 0.05132331674, 0.04902711698, 0.02301506314, 0.009558300975, 0.009117736768, 0.03582435837]
    x = [10, 20, 30, 40, 50, 60, 70]
    y = []
    y.append(np.asarray(origin))
    y.append(np.asarray(single_top))
    y.append(np.asarray(single_best))
    y.append(np.asarray(avg_top))
    y.append(np.asarray(avg_best))
    filename = 'Results/Error Rate V.S. Input Length_Richmond.png'
    title = 'OCR Error Rate v.s. Input Length for Richmond'
    ylabel = 'OCR Error Rate'
    xlabel = 'Input Length'
    xlim = [0, 80]
    ylim = [0, 0.6]
    lenlabel = ['OCR', 'Single-Top', 'Single-Best', 'Avg-Top', 'Avg-Best']
    # fig, ax = plt.subplots()
    index = np.asarray(x)
    bar_width = 1.2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c']
    rects = []
    for i in range(len(y)):
        rects.append(plt.bar(index - 6 + bar_width * i, y[i], bar_width,
                         alpha=opacity,
                         color=colors[i],
                         error_kw=error_config,
                         label=lenlabel[i]))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title)
    plt.xticks(index - bar_width * 2.5, [str(ele - 10) + '-' + str(ele) for ele in x])
    plt.legend(borderpad=2, bbox_transform=plt.gcf().transFigure, loc=2, fontsize=10)
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)


def plot_error_group_rich():
    origin = [3116, 6019, 7784, 30210, 123358, 22797, 80]
    x = [10, 20, 30, 40, 50, 60, 70]
    y = []
    y.append(np.asarray(origin))
    filename = 'Results/Number of Inputs V.S. Input Length_Richmond.png'
    title = 'Number of Inputs v.s. Input Length for Richmond'
    ylabel = '#Inputs'
    xlabel = 'Input Length'
    xlim = [0, 80]
    ylim = [0, 130000]
    lenlabel = ['Number of Inputs']
    # fig, ax = plt.subplots()
    index = np.asarray(x)
    bar_width = 2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c']
    rects = []
    for i in range(len(y)):
        rects.append(plt.bar(index - bar_width * 1.5 + bar_width * i, y[i], bar_width,
                         alpha=opacity,
                         color=colors[i],
                         error_kw=error_config,
                         label=lenlabel[i]))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title)
    plt.xticks(index - bar_width, [str(ele - 10) + '-' + str(ele) for ele in x])
    plt.legend(borderpad=2, bbox_transform=plt.gcf().transFigure, loc=2, fontsize=10)
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)

def plot_error_group():
    origin = [3057, 4208, 4538, 21251, 86248, 61174, 32031, 22325, 16288, 3500, 273, 8]
    x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    y = []
    y.append(np.asarray(origin))
    filename = 'Results/Number of Inputs V.S. Input Length_Book.png'
    title = 'Number of Inputs v.s. Input Length for Book'
    ylabel = '#Inputs'
    xlabel = 'Input Length'
    xlim = [0, 120]
    ylim = [0, 90000]
    lenlabel = ['Number of Inputs']
    # fig, ax = plt.subplots()
    index = np.asarray(x)
    bar_width = 2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c']
    rects = []
    for i in range(len(y)):
        rects.append(plt.bar(index - bar_width * 1.5 + bar_width * i, y[i], bar_width,
                         alpha=opacity,
                         color=colors[i],
                         error_kw=error_config,
                         label=lenlabel[i]))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title)
    plt.xticks(index - bar_width, [str(ele - 10) + '-' + str(ele) for ele in x])
    plt.legend(borderpad=2, bbox_transform=plt.gcf().transFigure, loc=2, fontsize=10)
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)

# plot_error_length()
# plot_error_length_rich()
# plot_error_group_rich()
plot_error_group()
