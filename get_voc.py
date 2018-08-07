import sys
import re
from os.path import join as pjoin

folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR/lm/char/'
folder_lm = sys.argv[1]
filename = pjoin(folder_data, folder_lm, 'train.text')
dict_voc = {}
for line in file(filename):
    for char in re.sub(r'[^\x00-\x7F]', ' ', line.strip('\n')):
        if char not in dict_voc:
            dict_voc[char] = 1
out_file = pjoin(folder_data, folder_lm, 'voc')
with open(out_file, 'w') as f_:
    for char in dict_voc:
        f_.write(char + '\n')
