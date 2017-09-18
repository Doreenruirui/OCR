from os.path import join, exists
import re

folder_multi = '/scratch/dong.r/Dataset/OCR/book'

with open(join(folder_multi, 'man_wit.s.y.txt'), 'w') as f_:
    for line in file(join(folder_multi, 'man_wit.y.txt')):
        cur_line = line.strip('\n').replace('\t', ' ')
        cur_line = ' '.join([ele for ele in cur_line.split(' ') if len(ele.strip()) > 0])
        cur_line = re.sub('\xc5\xbf', 's', cur_line)
        f_.write(cur_line + '\n')


with open(join(folder_multi, 'man.s.y.txt'), 'w') as f_:
    for line in file(join(folder_multi, 'man.y.txt')):
        cur_line = line.strip('\n').replace('\t', ' ')
        cur_line = ' '.join([ele for ele in cur_line.split(' ') if len(ele.strip()) > 0])
        cur_line = re.sub('\xc5\xbf', 's', cur_line)
        f_.write(cur_line + '\n')


