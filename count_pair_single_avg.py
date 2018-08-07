from levenshtein import count_pair
from multiprocessing import Pool
import sys
from os.path import join as pjoin
import re

folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR/'
#cur_folder = 'book/0/0/50/test_20/20/'
#cur_folder = 'multi/0/0/50/test/'
cur_folder = sys.argv[1]
file1 = sys.argv[2]
file2 = sys.argv[3]
out_file = sys.argv[4]
list_x = []
list_y = []
flag_x = int(sys.argv[5])
flag_y = int(sys.argv[6])
for line in file(pjoin(folder_data, cur_folder, file1)):
    if flag_x:
        line = re.sub(r'[^\x00-\x7F]', '', line.strip().lower())
    else:
        line = line.strip().lower()
    list_x.append(line)
for line in file(pjoin(folder_data, cur_folder, file2)):
    if flag_x:
        line = re.sub(r'[^\x00-\x7F]', '', line.strip().lower())
    else:
        line = line.strip().lower()
    list_y.append(line)

pool = Pool(100)
i1, d1, r1, list_op, list_op_str = count_pair(pool, list_x, list_y)
print i1, d1, r1
with open(pjoin(folder_data, cur_folder, 'op.' + out_file), 'w') as f_:
    for op in list_op:
        f_.write('\t'.join(map(str,op)) + '\n')
with open(pjoin(folder_data, cur_folder, 'str.' + out_file), 'w') as f_:
    for op_str in list_op_str:
        f_.write(op_str + '\n')

