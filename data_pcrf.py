# encoding: utf-8
import sys
from os.path import join
import re

def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

def convert_data(cur_folder, out_folder, cur_prefix):
    def convert(postfix, list_out):
        for line in file(join(cur_folder, cur_prefix + '.' + postfix + '.txt'), 'r'):
            line = re.sub('\xc5\xbf', 's', line)
            line = remove_nonascii(line.strip())
            new_line = ' '.join([ele for ele in line.replace('\t', ' ').split(' ') if len(ele.strip()) > 0])
            chars = [ele for ele in new_line]
            new_chars = []
            for char in chars:
                if char == '_':
                    new_chars.append('UNDERSCORE')
                elif char == ' ':
                    new_chars.append('_')
                else:
                    new_chars.append(char)
            list_out.append(' '.join(new_chars))
    cur_folder = join(folder_data, cur_folder)
    out_folder = join(folder_data, out_folder)
    list_x = []
    #list_y = []
    convert('x', list_x)
    #convert('y', list_y)
    num_line = len(list_x)
    with open(join(out_folder, cur_prefix + '.txt'), 'w') as f_out:
        for i in range(num_line):
            f_out.write(list_x[i] + '\n')
            #f_out.write(list_x[i] + '\t' + list_y[i] + '\n')


folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR'
arg_input = sys.argv[1]
arg_output = sys.argv[2]
arg_prefix = sys.argv[3]
convert_data(arg_input, arg_output, arg_prefix)







