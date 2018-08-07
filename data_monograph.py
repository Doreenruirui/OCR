from os.path import join, exists
import os
from os import listdir
import numpy as np


line_length = 40
folder_input = '/gss_gpfs_scratch/dong.r/Dataset/unprocessed/ICDAR2017/'
folder_dataset = ['eng_monograph', 'eng_periodical']
folder_out = {'eng_monograph': '/gss_gpfs_scratch/dong.r/Dataset/OCR/monograph/',
              'eng_periodical': '/gss_gpfs_scratch/dong.r/Dataset/OCR/periodical/'}


def split_data():
    for dataset in folder_dataset:
        pair_x = []
        pair_y = []
        folder_data = join(folder_input, dataset)
        cur_folder_out = join(folder_out[dataset])
        if not exists(cur_folder_out):
            os.makedirs(cur_folder_out)
        list_file = [ele for ele in listdir(folder_data) if not ele.startswith('.')]
        for fn in list_file:
            fpath = join(folder_data, fn)
            with open(fpath, 'r') as f_:
                lines = f_.readlines()
            ocr = lines[1][len('[OCR_aligned]'):].strip('\r\n').decode('utf-8')
            gs = lines[2][len('[ GS_aligned]'):].strip('\r\n').decode('utf-8')
            if len(ocr) != len(gs):
                print fn
            num_char = len(ocr)
            cur_num = 0
            while cur_num < num_char:
                cur_len = int(np.floor(np.random.random() * 20)) + line_length
                cur_len = min(cur_len, num_char - cur_num)
                new_x = ocr[cur_num: cur_num + cur_len]
                new_y = gs[cur_num: cur_num + cur_len]
                if '#' not in new_y:
                    new_x = new_x.replace('@', '').strip()
                    new_y = new_y.replace('@', '').strip()
                    new_x = ' '.join([ele for ele in new_x.split(' ') if len(ele.strip()) > 0])
                    new_y = ' '.join([ele for ele in new_y.split(' ') if len(ele.strip()) > 0])
                    pair_x.append(new_x)
                    pair_y.append(new_y)
                cur_num += cur_len
        with open(join(cur_folder_out, 'pair.x'), 'w') as f_:
            for pair in pair_x:
                f_.write(pair.encode('utf-8') + '\n')
        with open(join(cur_folder_out, 'pair.y'), 'w') as f_:
            for pair in pair_y:
                f_.write(pair.encode('utf-8') + '\n')


split_data()






