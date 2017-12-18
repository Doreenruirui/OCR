import sys
from os.path import join as pjoin
import os
from os.path import exists

folder_data = sys.argv[1]

list_x = []
list_y = []
num_wit = []
for line in file(pjoin(folder_data, 'man_wit.test.x.txt')):
    witness = [ele.strip() for ele in line.strip().split('\t') if len(ele.strip()) > 0]
    list_x.append(witness)
    num_wit.append(len(witness) - 1)
for line in file(pjoin(folder_data, 'man_wit.test.y.txt')):
    list_y.append(line.strip())
num_line = len(list_x)
num_all = 0
for i in range(50):
    index = []
    cur_folder = pjoin(folder_data, str(i))
    if not exists(cur_folder):
        os.makedirs(cur_folder)
    for j in range(num_line):
        if num_wit[j] >= i:
            index.append(j)
    print i, len(index)
    with open(pjoin(cur_folder, 'man_wit.test.x.txt'), 'w') as f_:
        for lid in index:
            f_.write('\t'.join(list_x[lid][:i + 1]) + '\n')
    with open(pjoin(cur_folder, 'man_wit.test.y.txt'), 'w') as f_:
        for lid in index:
            f_.write(list_y[lid] + '\n')
    num_all += len(index)
print num_all
