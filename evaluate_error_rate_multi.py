import os
from os.path import join as pjoin
from levenshtein import align_pair, align_one2many, align_beam, align
from multiprocessing import Pool
import numpy as np
import re
import sys
import plot_curve


# folder_data = '/Users/doreen/Documents/Experiment/dataset/OCR/'
folder_data = '/scratch/dong.r/Dataset/OCR'

def remove(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def error_rate(dis_xy, len_y):
    micro_error = 0
    len_x = len(dis_xy)
    num_emp = 0
    for i in range(len_x):
        if len_y[i] == 0:
            num_emp += 1
        else:
            micro_error += dis_xy[i] * 1. / (len_y[i])
    print num_emp
    micro_error = micro_error * 1. / (len_x - num_emp)
    macro_error = sum(dis_xy) * 1. / sum(len_y)
    return micro_error, macro_error



def evaluate_best(folder_name, out_folder, prefix='dev', group='1', beam_size=100, start=0, end=-1, flag_char=1):
    global folder_data
    folder_data = pjoin(folder_data, folder_name)
    if end == -1:
        file_name = pjoin(folder_data, out_folder, prefix + '.om' + group + '.txt')
    else:
        file_name = pjoin(folder_data, out_folder, prefix + '.om' + group + '.txt.' + str(start) + '_' + str(end))
    line_id = 0
    list_dec = []
    list_beam = []
    list_top = []
    for line in file(file_name):
        line_id += 1
        cur_str=line.split('\t')[0].strip().lower()
        if line_id % beam_size == 1:
            if len(list_beam) == beam_size:
                list_dec.append(list_beam)
                list_beam = []
            list_top.append(cur_str)
        list_beam.append(cur_str)
    list_dec.append(list_beam)
    with open(pjoin(folder_data, prefix + '.y.txt'), 'r') as f_:
        list_y = [ele.strip().lower() for ele in f_.readlines()][start:end]
    if flag_char:
        len_y = [len(y) for y in list_y]
    else:
        len_y = [len(y.split()) for y in list_y]
    print len(len_y)
    nthread=100
    P = Pool(nthread)
    dis_by, best_str = align_beam(P, list_y, list_dec, flag_char)
    dis_ty = align_pair(P, list_y,  list_top, flag_char)
    dis = np.asarray(zip(dis_by, dis_ty, len_y))
    if end == -1:
        if flag_char:
            outfile = pjoin(folder_data, out_folder, prefix + '.ec' + group + '.txt')
        else:
            outfile = pjoin(folder_data, out_folder, prefix + '.ew' + group + '.txt')
    else:
        if flag_char:
            outfile = pjoin(folder_data, out_folder, prefix + '.ec' + group + '.txt.' + str(start) + '_' + str(end))
        else:
            outfile = pjoin(folder_data, out_folder, prefix + '.ew' + group + '.txt.' + str(start) + '_' + str(end))
    np.savetxt(outfile, dis, fmt='%d')


def evaluate_group_ocr(folder_name, prefix='dev'):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    with open(pjoin(cur_folder_data, prefix + '.x.txt'), 'r') as f_:
        list_x = [ele.lower().strip('\n').split('\t')[0] for ele in f_.readlines()]
    with open(pjoin(cur_folder_data, prefix + '.y.txt'), 'r') as f_:
        list_y = [ele.strip().lower() for ele in f_.readlines()]
    len_yc = [len(y) for y in list_y]
    P = Pool(100)
    dis_xy = align_pair(P, list_x, list_y)
    np.savetxt(pjoin(cur_folder_data, prefix + '.ec.txt'), np.asarray(zip(dis_xy, len_yc)), fmt='%d')


def evaluate_all(folder_name, g1_file, g2_file, g3_file, ocr_file):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    dis1 = np.loadtxt(pjoin(cur_folder_data, g1_file), dtype=int)
    dis2 = np.loadtxt(pjoin(cur_folder_data, g2_file), dtype=int)
    dis3 = []
    for line in file(pjoin(cur_folder_data, g3_file)):
        items = line.strip('\n').split('\t')
        dis3.append([int(items[0]), int(items[1]), int(items[2]), int(items[-1])])
    dis3 = np.asarray(dis3)
    dis4 = np.loadtxt(pjoin(cur_folder_data, ocr_file), dtype=int)
    for i in range(dis1.shape[1] -1):
        micro, macro = error_rate(dis1[:,i], dis1[:, -1])
        print micro, macro
    for i in range(dis2.shape[1] -1):
        micro, macro = error_rate(dis2[:,i], dis2[:, -1])
        print micro, macro
    for i in range(dis3.shape[1] - 1):
        micro, macro = error_rate(dis3[:,i], dis3[:, -1])
        print micro, macro
    #micro, macro = error_rate(dis3, dis2[:, -1])
    #print micro, macro
    micro, macro = error_rate(dis4[:, 0], dis4[:, -1])
    print micro, macro


def evaluate_plot(folder_name, g1_file, g2_file, g3_file, ocr_file):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    dis1 = np.loadtxt(pjoin(cur_folder_data, g1_file), dtype=float)
    dis2 = np.loadtxt(pjoin(cur_folder_data, g2_file), dtype=float)
    dis4 = np.loadtxt(pjoin(cur_folder_data, ocr_file), dtype=float)
    dis3 = []
    num_input = []
    with open(pjoin(cur_folder_data, g3_file), 'r') as f_:
        lines = f_.readlines()
        for line in lines:
            items = line.strip('\n').split('\t')
            dis3.append([float(items[0]), float(items[1]), float(items[2]), float(items[-1])])
            num_input.append((len(items) - 1) * 1. /3)
    num_input = np.asarray(num_input)
    dis3 = np.asarray(dis3)
    dict_id2name = {0:'ocr', 1:'single_top', 2:'single_best', 3:'soft_top', 4:'soft_best', 5:'avg_top', 6:'avg_best'}
    dict_id2error = {}
    dict_id2error[6]  = dis1[:, 0] / dis1[:, -1]
    dict_id2error[5] = dis1[:, 1] / dis2[:, -1]
    dict_id2error[4] = dis2[:, 0] / dis2[:, -1]
    dict_id2error[3] = dis2[:, 1] / dis2[:, -1]
    dict_id2error[1] = dis3[:, 1] / dis3[:, -1]
    dict_id2error[2] = dis3[:, 2] / dis3[:, -1]
    dict_id2error[0] = dis4[:, 0] / dis4[:, -1]
    group1 = [i * 5 for i in range(20)]
    group2 = [i * 5 for i in range(1, 21)]
    group2[-1] = float('inf')
    group = zip(group1, group2)
    dict_id2error_group = {}
    for item in dict_id2error:
        dict_id2error_group[item] = {}
        for gp in group:
            dict_id2error_group[item][gp] = []
    for i in range(num_input.shape[0]):
        for s, e in group:
            if s < num_input[i] <= e:
                for key in dict_id2error_group:
                    dict_id2error_group[key][(s, e)].append(dict_id2error[key][i])
                break
    dict_id2error_group_avg = {}
    dict_id2error_group_num = {}
    for key in dict_id2error_group:
        dict_id2error_group_avg[key] = {}
        for gp in group:
            cur_num = len(dict_id2error_group[key][gp])
            dict_id2error_group_num[gp] = cur_num
            dict_id2error_group_avg[key][gp] = np.mean(np.asarray(dict_id2error_group[key][gp])) if cur_num > 0 else 0
    y = [[] for _ in range(len(dict_id2name))]
    for i in range(len(dict_id2name)):
        for gp in group:
            y[i].append(dict_id2error_group_avg[i][gp])
    # y.append([dict_id2error_group_num[gp] for gp in group])
    print [dict_id2error_group_num[gp] for gp in group]
    x = [ele[1] for ele in group]
    x[-1] = 100
    file = 'error_vs_number_of_inputs.png'
    title = 'Error rate on subgroup of different number of inputs'
    xlabel = 'number of inputs'
    ylabel = 'Error Rate'
    xlim = [0, 80]
    ylim = [0, 0.4]
    lenlabel = [dict_id2name[i] for i in range(len(dict_id2name))]
    fig = 1
    plot_curve.plot(x, y, xlabel,ylabel, xlim, ylim, lenlabel,title,fig,file)



def evaluate_plot_2(folder_name, g1_file, g2_file, g3_file, ocr_file):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    dis1 = np.loadtxt(pjoin(cur_folder_data, g1_file), dtype=float)
    dis2 = np.loadtxt(pjoin(cur_folder_data, g2_file), dtype=float)
    dis4 = np.loadtxt(pjoin(cur_folder_data, ocr_file), dtype=float)
    dis3 = []
    num_input = []
    with open(pjoin(cur_folder_data, g3_file), 'r') as f_:
        lines = f_.readlines()
        for line in lines:
            items = line.strip('\n').split('\t')
            dis3.append([float(items[0]), float(items[1]), float(items[2]), float(items[-1])])
            num_input.append((len(items) - 1) * 1. /3)
    num_input = np.asarray(num_input)
    print len([i for i in num_input if i == 1])
    dis3 = np.asarray(dis3)
    dict_id2name = {0:'ocr', 1:'single_top', 2:'single_best', 3:'soft_top', 4:'soft_best', 5:'avg_top', 6:'avg_best'}
    dict_id2error = {}
    dict_id2error[6]  = dis1[:, 0] / dis1[:, -1]
    dict_id2error[5] = dis1[:, 1] / dis1[:, -1]
    dict_id2error[4] = dis2[:, 0] / dis2[:, -1]
    dict_id2error[3] = dis2[:, 1] / dis2[:, -1]
    dict_id2error[1] = dis3[:, 1] / dis3[:, -1]
    dict_id2error[2] = dis3[:, 2] / dis3[:, -1]
    dict_id2error[0] = dis4[:, 0] / dis4[:, -1]
    group = [i for i in range(1, 101)]
    dict_id2error_group = {}
    for item in dict_id2error:
        dict_id2error_group[item] = {}
        for gp in group:
            dict_id2error_group[item][gp] = []
    for i in range(num_input.shape[0]):
        for n in group:
            if n == num_input[i]:
                for key in dict_id2error_group:
                    dict_id2error_group[key][n].append(dict_id2error[key][i])
    dict_id2error_group_avg = {}
    dict_id2error_group_num = {}
    for key in dict_id2error_group:
        dict_id2error_group_avg[key] = {}
        for gp in group:
            cur_num = len(dict_id2error_group[key][gp])
            dict_id2error_group_num[gp] = cur_num
            dict_id2error_group_avg[key][gp] = np.mean(np.asarray(dict_id2error_group[key][gp])) if cur_num > 0 else 0
    y = [[] for _ in range(len(dict_id2name))]
    for i in range(len(dict_id2name)):
        for gp in group:
            y[i].append(dict_id2error_group_avg[i][gp])
    # y.append([dict_id2error_group_num[gp] for gp in group])
    print [dict_id2error_group_num[gp] for gp in group][1:80]
    x = group
    file = 'error_vs_number_of_inputs_4.png'
    title = 'Error rate on subgroup of different number of inputs'
    xlabel = 'number of inputs'
    ylabel = 'Error Rate'
    xlim = [2, 20]
    ylim = [0, 0.4]
    lenlabel = [dict_id2name[i] for i in range(len(dict_id2name))]
    fig = 1
    plot_curve.plot(x, y, xlabel,ylabel, xlim, ylim, lenlabel,title,fig,file)



if True:
    cur_folder = sys.argv[1]
    cur_prefix = sys.argv[2]
    cur_out = sys.argv[3]
    beam = int(sys.argv[4])
    start_line = int(sys.argv[5])
    end_line = int(sys.argv[6])
    group_no = sys.argv[7]
    evaluate_best(cur_folder, cur_out, cur_prefix,  group=group_no, beam_size=beam, start=start_line, end=end_line, flag_char=1)
else:
    cur_folder = sys.argv[2]
    cur_prefix = sys.argv[3]
    evaluate_group_ocr(cur_folder, cur_prefix)
    evaluate_all('richmond/0/0/50/train_new', 'group.ec1.txt', 'group.ec2.txt', 'group.em3.txt', 'group.ec.txt')
    evaluate_plot_2('richmond/0/0/50/train_new', 'group.ec1.txt', 'group.ec2.txt', 'group.em3.txt', 'group.ec.txt')