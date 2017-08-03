import gzip
from os.path import join, exists
from os import listdir, makedirs
import json
from collections import OrderedDict
from multiprocessing import Pool


folder_data = '/scratch/dong.r/Dataset/unprocessed/witnesses.json'
folder_out = '/scratch/dong.r/Dataset/OCR/multi'


def process_line(line):
    dict_pair = OrderedDict()
    line = json.loads(line.strip('\r\n'))
    cur_id = line['id']
    lines = line['lines']
    for item in lines:
        begin = item['begin']
        text = item['text']
        if (cur_id, begin) not in dict_pair:
            dict_pair[(cur_id, begin)] = text
        if 'witness' in item:
            for wit in item['witness']:
                begin = wit['begin']
                cur_id = wit['id']
                text = wit['text']
                if (cur_id, begin) not in dict_pair:
                    dict_pair[(cur_id, begin)] = text
    return dict_pair


def process_file(paras):
    fn, out_fn = paras
    with gzip.open(join(folder_data, fn), 'r') as f_:
        content = f_.readlines()
    out_x = open(out_fn + '.x', 'w')
    out_y = open(out_fn + '.y', 'w')
    out_z = open(out_fn + '.z', 'w')
    out_z_info = open(out_fn + '.z.info', 'w')
    out_x_info = open(out_fn + '.x.info', 'w')
    out_y_info = open(out_fn + '.y.info', 'w')
    cur_line_no = 0
    cur_group = 0
    for line in content:
        line = json.loads(line.strip('\r\n'))
        cur_id = line['id']
        lines = line['lines']
        for item in lines:
            begin =  item['begin']
            text = item['text'].replace('\n', ' ')
            out_x.write(text.encode('utf-8') + '\n')
            wit_info = ''
            wit_str = ''
            man_str = ''
            man_info = ''
            num_manul = 0
            num_wit = 0
            if 'witnesses' in item:
                for wit in item['witnesses']:
                    wit_begin = wit['begin']
                    wit_id = wit['id']
                    wit_text = wit['text'].replace('\n', ' ')
                    if 'manual' not in wit_id:
                        num_wit += 1
                        if len(wit_str) == 0:
                            wit_info = str(wit_id) + '\t' + str(wit_begin)
                            wit_str = wit_text.encode('utf-8')
                        else:
                            wit_info += '\t' + str(wit_id) + '\t' + str(wit_begin)
                            wit_str += '\t' + wit_text.encode('utf-8')
                    else:
                        num_manul += 1
                        if len(man_str) == 0:
                            man_str = wit_text.encode('utf-8')
                            man_info = str(wit_id) + '\t' + str(wit_begin)
                        else:
                            man_str += '\t' + wit_text.encode('utf-8')
                            man_info += '\t' + str(wit_id) + '\t' + str(wit_begin)
            if len(man_str.strip()) > 0:
                out_z.write(man_str + '\n')
                out_z_info.write(str(cur_line_no) + '\t' + man_info + '\n')
            if len(wit_str.strip()) > 0:
                out_y.write(wit_str + '\n')
                out_y_info.write(str(cur_line_no) + '\t' +  wit_info + '\n')
            # out_y.write(wit_str + '\n')
            # out_y_info.write(wit_info + '\n')
            # out_z.write(man_str + '\n')
            # out_z_info.write(man_info + '\n')
            out_x_info.write(str(cur_group) + '\t' + str(cur_line_no) + '\t' + str(cur_id) + '\t' + str(begin) + '\t' + str(len(text) + begin) + '\t' + str(num_wit) + '\t' + str(num_manul) + '\n')
            cur_line_no += 1
        cur_group += 1

    out_x.close()
    out_y.close()
    out_z.close()
    out_x_info.close()
    out_y_info.close()
    out_z_info.close()


def merget_file():
    list_file = [ele for ele in listdir(folder_data) if ele.startswith('part')]
    list_out_file= [join(folder_out, 'pair.' + str(i)) for i in range(len(list_file))]
    out_fn = join(folder_out, 'pair')
    out_x = open(out_fn + '.x', 'w')
    out_y = open(out_fn + '.y', 'w')
    out_z = open(out_fn + '.z', 'w')
    out_z_info = open(out_fn + '.z.info', 'w')
    out_x_info = open(out_fn + '.x.info', 'w')
    out_y_info = open(out_fn + '.y.info', 'w')
    last_num_line = 0
    last_num_group = 0
    total_num_y = 0
    total_num_z = 0
    for fn in list_out_file:
        num_line = 0
        for line in file(fn + '.x'):
            out_x.write(line)
            num_line += 1
        for line in file(fn + '.y'):
            out_y.write(line)
        for line in file(fn + '.z'):
            out_z.write(line)
        dict_x2liney = {}
        dict_x2linez = {}
        for line in file(fn + '.y.info'):
            line = line.split('\t')
            line[0] = str(int(line[0]) + last_num_line)
            dict_x2liney[line[0]] = total_num_y
            total_num_y += 1
            out_y_info.write('\t'.join(line))
        for line in file(fn + '.z.info'):
            line = line.split('\t')
            line[0] = str(int(line[0]) + last_num_line)
            dict_x2linez[line[0]] = total_num_z
            total_num_z += 1
            out_z_info.write('\t'.join(line))
        num_group = 0
        for line in file(fn + '.x.info'):
            line = line.strip('\r\n').split('\t')
            cur_group = int(line[0])
            line[0] = str(int(line[0]) + last_num_group)
            line[1] = str(int(line[1]) + last_num_line)
            if line[1] in dict_x2liney:
                line.append(str(dict_x2liney[line[1]]))
            else:
                line.append('0')
            if line[1] in dict_x2linez:
                line.append(str(dict_x2linez[line[1]]))
            else:
                line.append('0')
            out_x_info.write('\t'.join(line))
            if cur_group > num_group:
                num_group = cur_group
        last_num_group += num_group
        last_num_line += num_line
    out_x.close()
    out_y.close()
    out_z.close()
    out_x_info.close()
    out_y_info.close()
    out_z_info.close()


def process_data():
    list_file = [ele for ele in listdir(folder_data) if ele.startswith('part')]
    list_out_file= [join(folder_out, 'pair.' + str(i)) for i in range(len(list_file))]
    if not exists(folder_out):
        makedirs(folder_out)
    # process_file((list_file[0], list_out_file[0]))
    pool = Pool(100)
    pool.map(process_file, zip(list_file, list_out_file))


# process_data()
merget_file()
