from os.path import join, exists
from collections import OrderedDict
import os
import re
from PyLib.operate_file import load_obj, save_obj


replace_xml = {'&lt;': '<', '&gt;': '>', '&quot;': '"',  '&apos;': '\'', '&amp;': '&'}


def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def split_data():
    dict_split = OrderedDict()
    dict_manual = OrderedDict()
    dict_wit = OrderedDict()
    total_man = 0
    total_wit = 0
    total_date_man_wit = 0
    total_date_man = 0
    total_date_wit = 0
    lid = 0
    for line in file(join(folder_multi, 'pair.x.info')):
        items = line.strip('\n').split('\t')
        cur_begin = int(items[3])
        cur_end = int(items[4])
        line_id = int(items[1])
        cur_id = items[2]
        cur_date = re.findall('[0-9]{4}-[0-9]{2}-[0-9]{2}', cur_id)
        cur_ed =  re.findall('ed-([0-9]{1})', cur_id)
        cur_seq = re.findall('seq-([0-9]{1})', cur_id)
        num_wit = int(items[5])
        num_manual = int(items[6])
        wit_line = -1
        if num_wit > 0:
            wit_line = int(items[7])
            total_wit += 1
        manual_line = -1
        if num_manual > 0:
            manual_line = int(items[8])
            total_man += 1
        print line_id
        if len(cur_date) > 0 and len(cur_seq) > 0 and len(cur_ed) > 0:
            if num_manual > 0 and num_wit > 0:
                if wit_line not in dict_split:
                    dict_split[wit_line] = []
                dict_split[wit_line].append((cur_date[0], cur_ed[0] + '-' + cur_seq[0], cur_begin, cur_end, line_id, manual_line, total_date_man_wit))
                total_date_man_wit += 1
            elif num_manual > 0:
                if line_id not in dict_manual:
                    dict_manual[line_id] = (cur_date[0], cur_ed[0] + '-' + cur_seq[0], cur_begin, cur_end, manual_line, total_date_man)
                total_date_man += 1
            elif num_wit > 0:
                if wit_line not in dict_wit:
                    dict_wit[wit_line] = []
                dict_wit[wit_line].append((cur_date[0], cur_ed[0] + '-' + cur_seq[0], cur_begin, cur_end, line_id, total_date_wit))
                total_date_wit += 1
        lid += 1
    save_obj(join(folder_multi, 'man_wit'), dict_split)
    save_obj(join(folder_multi, 'man'), dict_manual)
    save_obj(join(folder_multi, 'wit'), dict_wit)
    print total_man, total_wit, total_date_man_wit, total_date_man, total_date_wit


def write_manual():
    dict_manual = load_obj(join(folder_multi, 'man'))
    pair_z = []
    for line in file(join(folder_multi, 'pair.z')):
        pair_z.append(line.strip('\n'))
    line_id = 0
    out_x = open(join(folder_multi, 'man.x.txt'), 'w')
    out_y = open(join(folder_multi, 'man.y.txt'), 'w')
    out_info = open(join(folder_multi, 'man.info.txt'), 'w')
    for line in file(join(folder_multi, 'pair.x')):
        if line_id in dict_manual:
            for ele in replace_xml:
                line = re.sub(ele, replace_xml[ele], line)
            if len(remove_nonascii(line).strip()) > 0:
                cur_info = dict_manual[line_id]
                manul_line = cur_info[4]
                if len(remove_nonascii(pair_z[manul_line]).strip()) > 0 and '#' not in pair_z[manul_line]:
                    out_x.write(line)
                    out_y.write(pair_z[manul_line] + '\n')
                    out_info.write('\t'.join(map(str, cur_info[:-2])) + '\n')
        line_id += 1
    out_x.close()
    out_y.close()
    out_info.close()


def write_witness():
    dict_wit = load_obj(join(folder_multi, 'wit'))
    max_line = 0
    for witline in dict_wit:
        for item in dict_wit[witline]:
            if item[-2] > max_line:
                max_line = item[-2]
    print max_line
    num_line = max_line + 1
    pair_x = []
    for line in file(join(folder_multi, 'pair.x')):
        pair_x.append(line.strip('\n'))
    list_x = [None for _ in range(num_line)]
    list_info = [None for _ in range(num_line)]
    line_id = 0
    print num_line, len(list_x), len(pair_x)
    for line in file(join(folder_multi, 'pair.y')):
        if line_id in dict_wit:
            for info in dict_wit[line_id]:
                x_id = info[4]
                total_id = info[5]
                if len(remove_nonascii(pair_x[x_id]).strip()) > 0:
                    list_x[total_id] = pair_x[x_id] + '\t' + line.strip('\n')
                else:
                    list_x[total_id] = ''
                list_info[total_id] = info[:4]
        line_id += 1
    out_x = open(join(folder_multi, 'wit.x.txt'), 'w')
    out_info = open(join(folder_multi,'wit.info.txt'), 'w')
    for i in range(num_line):
        cur_x = list_x[i]
        cur_info = list_info[i]
        if len(cur_x) > 0:
            out_x.write(cur_x + '\n')
            out_info.write('\t'.join(map(str, cur_info)) + '\n')
    out_x.close()
    out_info.close()


def write_man_wit():
    dict_split = load_obj(join(folder_multi, 'man_wit'))
    max_line = 0
    for witline in dict_split:
        for item in dict_split[witline]:
            if item[-1] > max_line:
                max_line = item[-1]
    print max_line
    num_line = max_line + 1
    pair_x = []
    for line in file(join(folder_multi, 'pair.x')):
        pair_x.append(line.strip('\n'))
    pair_z = []
    for line in file(join(folder_multi, 'pair.z')):
        pair_z.append(line.strip('\n'))
    list_x = [None for _ in range(num_line)]
    list_y = [None for _ in range(num_line)]
    list_info = [None for _ in range(num_line)]
    line_id = 0
    print num_line, len(list_x), len(pair_x)
    for line in file(join(folder_multi, 'pair.y')):
        print line_id
        if line_id in dict_split:
            for info in dict_split[line_id]:
                x_id = info[4]
                z_id = info[5]
                total_id = info[6]
                print x_id
                cur_x = pair_x[x_id]
                for ele in replace_xml:
                    cur_x = re.sub(ele, replace_xml[ele], cur_x)
                cur_z = pair_z[z_id]
                for ele in replace_xml:
                    cur_z = re.sub(ele, replace_xml[ele], cur_z)
                if len(remove_nonascii(cur_x).strip()) > 0 and len(remove_nonascii(cur_z).strip()) > 0 and '#' not in cur_z:
                    for ele in replace_xml:
                        line = re.sub(ele, replace_xml[ele], line)
                    list_x[total_id] = cur_x + '\t' + line.strip('\n')
                else:
                    list_x[total_id] = ''
                list_y[total_id] = cur_z
                list_info[total_id] = list(info[:4])
        line_id += 1
    out_x = open(join(folder_multi, 'man_wit.x.txt'), 'w')
    out_y = open(join(folder_multi, 'man_wit.y.txt'), 'w')
    out_info = open(join(folder_multi, 'man_wit.info.txt'), 'w')
    for i in range(num_line):
        cur_x = list_x[i]
        cur_y = list_y[i]
        cur_info = list_info[i]
        print cur_info
        if len(cur_x) > 0:
            out_x.write(cur_x + '\n')
            out_y.write(cur_y + '\n')
            out_info.write('\t'.join(map(str, cur_info)) + '\n')
    out_x.close()
    out_y.close()
    out_info.close()


folder_multi = '/scratch/dong.r/Dataset/OCR/multi'

train_ratio = 0.8
tid = 0
sid = 0
# split_data()
write_manual()
# write_witness()
write_man_wit()
