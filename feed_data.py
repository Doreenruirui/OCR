import re
import numpy as np
from levenshtein import align_re
import sys
import json

###TO DO: Remove Non-ascii
###TO DO: Replace '-' with '_'
###TO DO: Replace ''' with '_'
###TO DO: Align the witnesses with target

def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

def recover(str1, str2):
    if len(str1) == 0:
        return '-' * len(str2), str2
    elif len(str2) == 0:
        return str1, '-' * len(str1)
    _, d, op = align_re(str1, str2)
    len1, len2 = d.shape
    len1 -= 1
    len2 -= 1
    # print(d[len1, len2])
    j = len2
    i = len1
    path = []
    while j >= 1 or i >= 1:
        path.append((i, j))
        if op[i, j] == 1:
            j -= 1
        elif op[i, j] == 2:
            i -= 1
        elif op[i, j] == 3:
            i -= 1
            j -= 1
        else:
            i -= 1
            j -= 1
    path = path[::-1]
    res1 = ''
    res2 = ''
    for (i, j) in path:
        char1 = str1[i - 1]
        char2 = str2[j - 1]
        if op[i, j] == 1:
            res1 += '-'
            res2 += char2
        elif op[i, j] == 2:
            res2 += '-'
            res1 += char1
        else:
            res1 += char1
            res2 += char2
    return res1, res2

def prepare_data(filename):
    cluster_id = 0
    id = 0
    num_char = 0
    # remove nonascii
    f_ = open(filename + '.new', 'w')
    for line in file(filename):
        line = re.sub('-', '_', line)
        line = re.sub('`', '_', line)
        items = [ele.strip() for ele in line.strip().split('\t') if len(ele.strip()) > 0]
        target = items[0]
        if len(items) < 2:
            print line
            f_.write(str(cluster_id) + '\t' + str(id) + '\t' + '1863-01-03' + '\t' + '0' + '\t' + str(
                len(target)) + '\t' + target + '\n')
            id += 1
            f_.write(str(cluster_id) + '\t' + str(id) + '\t' + '1863-01-03' + '\t' + '0' + '\t' + str(
                len(target)) + '\t' + target + '\n')
            id += 1
        else:
            for wit in items[1:]:
                wit = remove_nonascii(wit)
                t1, w1 = recover(target, wit)
                f_.write(str(cluster_id) + '\t' + str(id) + '\t' + '1863-01-03' + '\t' + '0' + '\t' + str(len(wit)) + '\t' + w1 + '\n')
                id += 1
                f_.write(str(cluster_id) + '\t' + str(id) + '\t' + '1863-01-03' + '\t' + '0' + '\t' + str(len(target)) + '\t' + t1 + '\n')
                id += 1
        cluster_id += 1

    f_.close()

def prepare_target(filename):
    cluster_id = 0
    id = 0
    num_char = 0
    # remove nonascii
    f_ = open(filename + '.new', 'w')
    f_1 = open(filename + '.mv', 'w')
    for line in file(filename):
        line = re.sub('-', '_', line)
        line = re.sub('`', '_', line)
        line = line.strip()
        dict_line = {}
        dict_line['cluster_id'] = cluster_id
        dict_line['id'] = id
        cluster_id += 1
        id += 1
        dict_line['date'] = '1863-01-03'
        dict_line['text'] = line
        f_.write(json.dumps(dict_line) + '\n')
        f_1.write(line + '\n')
    f_.close()

# prepare_data(sys.argv[1])
prepare_target(sys.argv[1])
