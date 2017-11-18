import string
import sys


def remove_empty_line(filename):
    empty = 0
    with open(filename + '.nonempty', 'w') as f_:
        for line in file(filename):
            line = line.strip('\n').translate(None, string.punctuation)
            line = ' '.join([ele for ele in line.split(' ') if len(ele) > 0])
            if len(line) <= 2:
                empty += 1
            else:
                f_.write(line + '\n')
    print empty


remove_empty_line(sys.argv[1])
    
