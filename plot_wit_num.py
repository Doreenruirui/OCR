import numpy as np
from plot_curve import plot
from os.path import join
import matplotlib.pyplot as plt


def main(name_error, name_avg):
    folder_data = '/home/rui/Dataset/OCR/richmond/single/0/0/test_10'
    training = {'Seq2Seq-Syn': 'nyt_0.3',
                'Seq2Seq-Noisy': 'noisy_100_40',
                'Seq2Seq-Boots': 'nyt_0.3_bootstrap',
                'Seq2Seq-Super': '40'}
    dict_evl = {}
    training_label = ['Seq2Seq-Syn', 'Seq2Seq-Noisy', 'Seq2Seq-Boots', 'Seq2Seq-Super']
    num_wit = []
    oc = []
    ow = []
    for line in file(join(folder_data, 'man_wit.test.ec.txt')):
        items = map(float, line.strip().split())
        num_wit.append(min(50, len(items[1:-1])))
        oc.append([min(items[:-1]), items[0], items[-1]])
    oc = np.asarray(oc)
    for line in file(join(folder_data, 'man_wit.test.ew.txt')):
        items = map(float, line.strip().split())
        ow.append([min(items[:-1]), items[0], items[-1]])
    ow = np.asarray(ow)
    for label in training_label:
        dict_evl[label] = []
        if label == 'OCR':
            dict_evl[label].append(oc)
            dict_evl[label].append(ow)
        else:
            cur_folder = training[label]
            dict_evl[label].append(np.loadtxt(join(folder_data, cur_folder, 'man_wit.test.avg.ec.txt')))
            dict_evl[label].append(np.loadtxt(join(folder_data, cur_folder, 'man_wit.test.avg.ew.txt')))
    num_group = 11
    group = np.arange(num_group)
    nline = len(num_wit)
    dict_error = {}
    for label in training_label:
        dict_error[label] = [[] for _ in range(num_group)]
        a = dict_evl[label]
        for i in range(nline):
            for j in range(1, num_group):
                if num_wit[i] == group[j]:
                    if 'CER' in name_error:
                        if name_error == 'CER':
                            dict_error[label][j].append([a[0][i, 1], a[0][i, -1]])
                        else:
                            dict_error[label][j].append([a[0][i, 0], a[0][i, -1]])
                    else:
                        if name_error == 'WER':
                            dict_error[label][j].append([a[1][i, 1], a[1][i, -1]])
                        else:
                            dict_error[label][j].append([a[1][i, 0], a[1][i, -1]])
    dict_error_avg = {}
    for label in training_label:
        dict_error_avg[label] = [0 for _ in range(num_group)]
        for j in range(1, num_group):
            cur_array = np.asarray(dict_error[label][j])
            print j, cur_array.shape
            if name_avg == 'micro-avg':
                dict_error_avg[label][j] = np.sum(cur_array[:, 0]) / np.sum(cur_array[:,1])
            else:
                dict_error_avg[label][j] = np.mean(cur_array[:, 0] / cur_array[:, 1])
    plot_error(name_avg, name_error, dict_error_avg, num_group, training_label)


def plot_error(name_avg, name_error, dict_error, ngroup, training_label):
    x = [ele for ele in range(1, ngroup)]
    y = []
    lenlabel = training_label
    for ele in training_label:
        y.append(dict_error[ele][1:])
    #if ele != 'OCR':
        	#y.append(np.asarray(dict_error['OCR'][1:-1]) - np.asarray(dict_error[ele][1:-1]))
    filename = 'Results/Error_Wit_%s.png' % name_error
    ylabel = '%s (%s)' % (name_error, name_avg)
    xlabel = 'Number of Witnesses'
    xlim = [0.199, 11.001]
    ylim = [0.0, 0.1]
    # plt.figure(figsize=(8, 10))
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(np.arange(10) * 1 + 2, [str(ele) for ele in np.arange(10) * 5 + 2], fontsize=15)
    plt.yticks(fontsize=20)
    colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c']
    cs = [ele + '-' for ele in colors]
    for i in range(len(y)):
        plt.plot(x, y[i], cs[i], label=lenlabel[i], linewidth=2)
        plt.legend(borderpad=0.5, bbox_transform=plt.gcf().transFigure, loc=2, fontsize=15)
    plt.tight_layout()
    plt.savefig(filename)

def main2(name_error, name_avg):
    folder_data = '/home/rui/Dataset/OCR/book1800/single/0/0/test_10'
    # a = np.loadtxt(join(folder_data, 'lm_nyt_low', 'man_wit.test.avg.ec.txt'))
    # b = np.loadtxt(join(folder_data, 'lm_nyt_low', 'man_wit.test.avg.ew.txt'))
    # c = np.loadtxt(join(folder_data, 'nyt_0.3', 'man.test.ec.txt'))
    # d = np.loadtxt(join(folder_data, 'nyt_0.3', 'man.test.ew.txt'))
    # dict_error['CER'][0] = c[:, np.asarray([1,2])]
    # dict_error['LCER'][0] = c[:, np.asarray([0,2])]
    # dict_error['WER'][0] = d[:, np.asarray([1,2])]
    # dict_error['LWER'][0] = d[:, np.asarray([0,2])]
    training = {'Seq2Seq-Syn': 'nyt_0.3',
                'Seq2Seq-Noisy': 'noisy_100_40',
                'Seq2Seq-Boots': 'nyt_0.3_bootstrap',
                'Seq2Seq-Super': '40'}
    dict_evl = {}
    training_label = ['Seq2Seq-Syn', 'Seq2Seq-Noisy', 'Seq2Seq-Boots', 'Seq2Seq-Super']
    for label in training_label:
        dict_evl[label] = []
        cur_folder = training[label]
        dict_evl[label].append(np.loadtxt(join(folder_data, cur_folder, 'man_wit.test.avg.ec.txt')))
        dict_evl[label].append(np.loadtxt(join(folder_data, cur_folder, 'man_wit.test.avg.ew.txt')))
    num_wit = []
    for line in file(join(folder_data, 'man_wit.test.ec.txt')):
        items = line.strip().split()
        num_wit.append(min(50, len(items[1:-1])))
    num_wit = num_wit
    num_group = 10
    group = np.arange(num_group) * 5
    nline = len(num_wit)
    dict_error = {}
    for label in training_label:
        dict_error[label] = [[] for _ in range(num_group)]
        a = dict_evl[label]
        for i in range(nline):
            for j in range(num_group - 1):
                if group[j] < num_wit[i] <= group[j + 1]:
                    if 'CER' in name_error:
                        if name_error == 'CER':
                            dict_error[label][j].append([a[0][i, 1], a[0][i, -1]])
                        else:
                            dict_error[label][j].append([a[0][i, 0], a[0][i, -1]])
                    else:
                        if name_error == 'WER':
                            dict_error[label][j].append([a[1][i, 1], a[1][i, -1]])
                        else:
                            dict_error[label][j].append([a[1][i, 0], a[1][i, -1]])
    # dict_error_new = {m: {ele: [0 for _ in range(num_group)] for ele in ['CER', 'WER', 'LCER', 'LWER']} for m in ['micro-avg', 'macro-avg']}
    dict_error_avg = {}
    for label in training_label:
        dict_error_avg[label] = [0 for _ in range(num_group)]
        for j in range(num_group - 1):
            cur_array = np.asarray(dict_error[label][j])
            print j, cur_array.shape
            if name_avg == 'micro-avg':
                dict_error_avg[label][j] = np.sum(cur_array[:, 0]) / np.sum(cur_array[:,1])
            else:
                dict_error_avg[label][j] = np.mean(cur_array[:, 0] / cur_array[:,1])
    plot_wit_error(name_avg, name_error, dict_error_avg, num_group, training_label)
    # plot_error(name_avg, name_error, dict_error_avg, num_group, training_label)
    # plot_error(name_error, 'WER', dict_error_new[name_error], num_group)
    # name_error = 'macro-avg'
    # plot_error(name_error, 'CER', dict_error_new[name_error], num_group)
    # plot_error(name_error, 'WER', dict_error_new[name_error], num_group)



def plot_wit_error(name_avg, name_error, dict_error, ngroup, training_label):
    x = np.arange(ngroup) * 5
    xstickers = ['%d-%d'%(x[i], x[i + 1]) for i in range(ngroup-1)]
    y = []
    lenlabel = training_label
    for ele in training_label:
        y.append(dict_error[ele])
    filename = 'Results/Book_Error_Wit_%s_%s.png' % (name_avg, name_error)
    ylabel = 'Error Rate'
    xlabel = 'Range of Number of Witnesses'
    xlim = [0, 50]
    ylim = [0.06, 0.16]
    index = np.asarray(x)
    bar_width = 0.2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c']
    rects = []
    for i in range(len(y)):
        rects.append(plt.bar(index - bar_width * 2 + bar_width * i, y[i], bar_width,
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
    # plt.title(title)
    plt.xticks(index - bar_width * 2, xstickers)
    plt.legend(borderpad=1, bbox_transform=plt.gcf().transFigure, loc=1, fontsize=10)
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)

main('CER', 'macro-avg')
