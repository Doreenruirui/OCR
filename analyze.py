import numpy as np
from os.path import join as pjoin
import sys
import plot_curve


folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR/'


def error_rate(dis_xy, len_y):
    micro_error = np.mean(dis_xy/len_y)
    macro_error = np.sum(dis_xy) / np.sum(len_y)
    return micro_error, macro_error


def evaluate_error(file_name, col):
    global folder_data
    group = [0, 0.1, 0.2, 0.3, 0.4, 0.5, float('inf')]
    dict_error = {}
    dict_origin = {}
    for ele in group[1:]:
        dict_error[ele] = []
        dict_origin[ele] = []
    for line in file(file_name):
        items = map(float, line.strip('\n').split(' '))
        cur_error = items[-2] / items[-1]
        for i in range(1, 7):
            if group[i - 1] <= cur_error < group[i]:
                dict_error[group[i]].append(items[col]/items[-1])
                dict_origin[group[i]].append(items[-2]/items[-1])
                break
    for ele in dict_error:
        print ele, np.mean(dict_error[ele]), np.mean(dict_origin[ele])


def evaluate_all():
    global folder_data
    folder_name = 'richmond/0/0/50/train_new'
    g1_file = 'man_wit.test.ec1.txt'
    g2_file = 'man_wit.test.ec2.txt'
    g3_file = 'man_wit.test.ec3.txt'
    g4_file = 'man_wit.test.richmond.ec4.txt'
    g5_file = 'man_wit.test.richmond.ec.txt'
    # g6_file = 'man_wit_rev.test.ec.txt'
    ocr_file = 'man_wit.test.ec.txt'
    cur_folder = pjoin(folder_data, folder_name)
    dis0 = np.loadtxt(pjoin(cur_folder, ocr_file))
    dis1 = np.loadtxt(pjoin(cur_folder, g1_file))
    dis2 = np.loadtxt(pjoin(cur_folder, g2_file))
    dis3 = np.loadtxt(pjoin(cur_folder, g3_file))
    dis4 = np.loadtxt(pjoin(cur_folder, g4_file))
    dis5 = np.loadtxt(pjoin(cur_folder, g5_file))
    # dis6 = np.loadtxt(pjoin(cur_folder, g6_file))
    # dis3 = []
    # for line in file(pjoin(cur_folder, g3_file)):
    #     items = line.strip('\n').split('\t')
    #     dis3.append([float(items[0]), float(items[1]), float(items[2]), float(items[-1])])
    # dis3 = np.asarray(dis3)
    # dis4 = []
    # for line in file(pjoin(cur_folder, g4_file)):
    #     items = line.strip('\n').split('\t')
    #     dis4.append(float(items[0]))
    # dis4 =  np.asarray(dis4)
    micro, macro = error_rate(dis0[:,0], dis0[:,-1])
    print 'OCR', micro, macro
    micro, macro = error_rate(dis3[:, 0], dis1[:, -1])
    print 'OCR', micro, macro
    micro, macro = error_rate(dis5[:, 2], dis5[:, -1])
    print 'OCR', micro, macro
    micro, macro = error_rate(dis1[:,0], dis1[:, -1])
    print 'Average', 'Best', micro, macro
    micro, macro = error_rate(dis1[:,1], dis1[:, -1])
    print 'Average', 'Top', micro, macro
    micro, macro = error_rate(dis2[:,0], dis2[:, -1])
    print 'Softmax', 'Best', micro, macro
    micro, macro = error_rate(dis2[:,1], dis2[:, -1])
    print 'Softmax', 'Top', micro, macro
    micro, macro = error_rate(dis3[:,2], dis3[:, -1])
    print 'Single', 'Best', micro, macro
    micro, macro = error_rate(dis3[:,1], dis3[:, -1])
    print 'Single', 'Top', micro, macro
    micro, macro = error_rate(dis4[:,0], dis4[:, -1])
    print 'Lm', 'Richmond', 'Top', micro, macro
    micro, macro = error_rate(dis4[:,1], dis4[:, -1])
    print 'Lm', 'Richmond', 'Lower', micro, macro
    micro, macro = error_rate(dis5[:,0], dis5[:, -1])
    print 'Lm', 'Richmond', 'Decode the most clean', 'Best', micro, macro
    micro, macro = error_rate(dis5[:,1], dis5[:, -1])
    print 'Lm', 'Richmond', 'Decode the most clean', 'Top',  micro, macro


def evaluate_plot(folder_name, g1_file, g2_file, g3_file, g4_file, ocr_file):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    dis0 = np.loadtxt(pjoin(cur_folder_data, ocr_file), dtype=float)
    dis1 = np.loadtxt(pjoin(cur_folder_data, g1_file), dtype=float)
    dis2 = np.loadtxt(pjoin(cur_folder_data, g2_file), dtype=float)
    dis3 = np.loadtxt(pjoin(cur_folder_data, g3_file), dtype=float)
    dis4 = []
    with open(pjoin(cur_folder_data, ocr_file), 'r') as f_:
        for line in f_.readlines():
            items = line.strip('\n').split('\t')
            dis4.append(float(items[0]))
    dis4 =  np.asarray(dis4)
    num_input = []
    with open(pjoin(cur_folder_data, g4_file), 'r') as f_:
        lines = f_.readlines()
        for line in lines:
            items = line.strip('\n').split('\t')
            num_input.append((len(items) - 1) * 1. / 2)
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



def evaluate_plot_2(folder_name, g1_file, g2_file, g3_file, g4_file, g5_file, ocr_file):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    dis0 = []
    with open(pjoin(cur_folder_data, ocr_file), 'r') as f_:
        for line in f_.readlines():
            items = line.strip('\n').split('\t')
            dis0.append([float(items[0]), float(items[-1])])
    dis0 = np.asarray(dis0)
    dis1 = np.loadtxt(pjoin(cur_folder_data, g1_file), dtype=float)  #Average
    dis2 = np.loadtxt(pjoin(cur_folder_data, g2_file), dtype=float)  #Soft
    dis3 = np.loadtxt(pjoin(cur_folder_data, g3_file), dtype=float)  #Single
    dis4 = np.loadtxt(pjoin(cur_folder_data, g4_file), dtype=float)  #LM Rank
    dis5 = np.loadtxt(pjoin(cur_folder_data, g5_file), dtype=float)  #LM
    num_input = []
    with open(pjoin(cur_folder_data, ocr_file), 'r') as f_:
        lines = f_.readlines()
        for line in lines:
            items = line.strip('\n').split('\t')
            num_input.append((len(items) - 1))
    num_input = np.asarray(num_input)
    dict_id2name = {0:'ocr', 1:'single_top', 2:'single_best', 3:'soft_top', 4:'soft_best', 5:'avg_top', 6:'avg_best', 7: 'lm_top', 8: 'lm_best', 9: 'lm_rank'}
    dict_id2error = {}
    dict_id2error[6]  = dis1[:, 0] / dis1[:, -1]
    dict_id2error[5] = dis1[:, 1] / dis1[:, -1]
    dict_id2error[4] = dis2[:, 0] / dis2[:, -1]
    dict_id2error[3] = dis2[:, 1] / dis2[:, -1]
    dict_id2error[1] = dis3[:, 1] / dis3[:, -1]
    dict_id2error[2] = dis3[:, 2] / dis3[:, -1]
    dict_id2error[0] = dis0[:, 0] / dis0[:, -1]
    dict_id2error[8] = dis5[:, 0] / dis5[:, -1]
    dict_id2error[7] = dis5[:, 1] / dis5[:, -1]
    dict_id2error[9] = dis4[:, 0] / dis4[:, -1]
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
    plot_curve.plot(x, y, xlabel,ylabel, xlim, ylim, lenlabel, title, fig, file)

# evaluate_all()
evaluate_plot_2('multi/0/0/50/test',
                'man_wit.test.avg.ec.txt',
                'man_wit.test.soft.ec.txt',
                'man_wit.test.single.ec.txt',
                'man_wit.test.richmond.dec.ec.txt',
                'man_wit.test.richmond.ec.txt',
                'man_wit.test.ec.txt')
