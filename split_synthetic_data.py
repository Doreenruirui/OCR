import numpy as np
import sys


def split(filename):
    with open(filename + '.line', 'w') as f_:
        for line in file(filename):    
            line = line.strip('\n')
            num_char = 0
            len_line = len(line)
            while num_char <= len_line:
                cur_num = np.random.randn() * 5 + 45
                cur_num = max(70, cur_num)
                cur_num = min(0, cur_num)
                cur_num = min(cur_num, len_line - num_char)
                cur_str = line[num_char : num_char + cur_num]
                f_.write(cur_str + '\n')
                num_char += cur_num

split(sys.argv[1])

        
