import numpy as np
# from multiprocessing import Pool

output = None
output_str = None


def align(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    if len1 == 0:
        return 0
    if len2 == 0:
        return 0
    d = np.ones((len1 + 1, len2 + 1), dtype=int) * 1000000
    op = np.zeros((len1 + 1, len2 + 1), dtype=int)
    for i in range(len1 + 1):
        d[i, 0] = i
        op[i, 0] = 2
    for j in range(len2 + 1):
        d[0, j] = j
        op[0, j] = 1
    op[0, 0] = 0
    for i in range(1, len1 + 1):
        char1 = str1[i - 1]
        for j in range(1, len2 + 1):
            char2 = str2[j - 1]
            if char1 == char2:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i, j - 1] + 1, d[i - 1, j] + 1, d[i - 1, j - 1] + 1)
                if d[i, j] == d[i, j - 1] + 1:
                    op[i, j] = 1
                elif d[i, j] == d[i - 1, j] + 1:
                    op[i, j] = 2
                elif d[i, j] == d[i - 1, j - 1] + 1:
                    op[i, j] = 3
    return d[len1, len2]


def align_re(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    if len1 == 0:
        return 0
    if len2 == 0:
        return 0
    d = np.ones((len1 + 1, len2 + 1), dtype=int) * 1000000
    op = np.zeros((len1 + 1, len2 + 1), dtype=int)
    for i in range(len1 + 1):
        d[i, 0] = i
        op[i, 0] = 2
    for j in range(len2 + 1):
        d[0, j] = j
        op[0, j] = 1
    op[0, 0] = 0
    for i in range(1, len1 + 1):
        char1 = str1[i - 1]
        for j in range(1, len2 + 1):
            char2 = str2[j - 1]
            if char1 == char2:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i, j - 1] + 1, d[i - 1, j] + 1, d[i - 1, j - 1] + 1)
                if d[i, j] == d[i, j - 1] + 1:
                    op[i, j] = 1
                elif d[i, j] == d[i - 1, j] + 1:
                    op[i, j] = 2
                elif d[i, j] == d[i - 1, j - 1] + 1:
                    op[i, j] = 3
    return d[len1, len2], d, op


def recover(str1, str2):
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
    res_op = {}
    begin = 0
    end = 0
    middle = 0
    for (i, j) in path:
        char1 = str1[i - 1]
        char2 = str2[j - 1]
        if op[i, j] > 0:
            if i - 1 > len1 * 0.75:
                end += 1
            elif i - 1 < len1 * 0.25:
                begin += 1
            else:
                middle += 1
        if op[i, j] == 1:
            if 'eps' not in res_op:
                res_op['eps'] = {}
            res_op['eps'][char2] = res_op['eps'].get(char2, 0) + 1
        elif op[i, j] == 2:
            if char1 not in res_op:
                res_op[char1] = {}
            res_op[char1]['eps'] = res_op[char1].get('eps', 0) + 1
        elif op[i, j] == 3:
            if char1 not in res_op:
                res_op[char1] = {}
            res_op[char1][char2] = res_op[char1].get(char2, 0) + 1
        else:
            if char1 not in res_op:
                res_op[char1] = {}
            res_op[char1][char1] = res_op[char1].get(char1, 0) + 1
    return res_op, begin, middle, end


def align_one2many_thread(para):
    str1, list_str, thread_num, flag_char = para
    min_dis = float('inf')
    min_str = ''
    for i in range(len(list_str)):
        if not flag_char:
            dis = align(str1.split(), list_str[i].split())
        else:
            dis = align(str1, list_str[i])
        if dis < min_dis:
            min_dis = dis
            min_str = list_str[i]
    return thread_num, min_dis, min_str


def align_all(P, truth, cands, nthread, flag_char=1):
    global output, output_str
    output = np.zeros(nthread)
    output_str = ['' for i in range(nthread)]
    ncand = len(cands)
    ncand_thread = int(np.floor(ncand * 1. / (nthread - 1)))
    paras = [(truth,
              cands[x * ncand_thread: min((x + 1) * ncand_thread, ncand)],
              x, flag_char)
             for x in range(nthread)]

    results = P.map(align_one2many_thread, paras)
    # print results
    min_dis = float('inf')
    min_str = ''
    # print len(results)
    dict_res = {}
    for i in range(nthread):
        cur_thread, cur_dis, cur_str = results[i]
        if cur_thread not in dict_res:
            dict_res[cur_thread] = 1
        else:
            raise 'Conflicted Threads'
        if cur_dis < min_dis:
            min_dis = cur_dis
            min_str = cur_str
    return min_dis, min_str


def dis_thread(para):
    list_str1, list_str2, thread_num, flag_char = para
    cur_dis = np.zeros(len(list_str1), dtype=int)
    for i in range(len(list_str1)):
        str1 = list_str1[i]
        str2 = list_str2[i]
        if not flag_char:
            dis = align(str1.split(), str2.split())
        else:
            dis = align(str1, str2)
        cur_dis[i] = dis
    return thread_num, cur_dis


def recover_thread(para):
    list_str1, list_str2, thread_num = para
    res_op = {}
    begin = 0
    end = 0
    middle = 0
    for i in range(len(list_str1)):
        str1 = list_str1[i]
        str2 = list_str2[i]
        cur_op, cur_b, cur_m, cur_e = recover(str1.strip(), str2.strip())
        for ele in cur_op:
            if ele not in res_op:
                res_op[ele] = {}
            for k in cur_op[ele]:
                res_op[ele][k] = res_op[ele].get(k, 0) + cur_op[ele][k]
        begin += cur_b
        middle += cur_m
        end += cur_e
    return thread_num, res_op, begin, middle, end


def align_pair(P, truth, cands, nthread, flag_char=1):
    global output, output_str
    output = np.zeros(nthread)
    output_str = ['' for i in range(nthread)]
    ncand = len(cands)
    ncand_thread = int(np.floor(ncand * 1. / (nthread - 1)))
    paras = [(truth[x * ncand_thread: min((x + 1) * ncand_thread, ncand)],
              cands[x * ncand_thread: min((x + 1) * ncand_thread, ncand)],
              x, flag_char)
             for x in range(nthread)]

    results = P.map(dis_thread, paras)
    res = np.zeros(len(truth), dtype=int)
    for i in range(nthread):
        thread_num, list_dis = results[i]
        res[thread_num * ncand_thread: min((thread_num + 1) * ncand_thread, ncand)] = list_dis
    return res


def recover_pair(P, truth, cands, nthread):
    global output, output_str
    output = np.zeros(nthread)
    output_str = ['' for i in range(nthread)]
    ncand = len(cands)
    ncand_thread = int(np.floor(ncand * 1. / (nthread - 1)))
    paras = [(truth[x * ncand_thread: min((x + 1) * ncand_thread, ncand)],
              cands[x * ncand_thread: min((x + 1) * ncand_thread, ncand)],
              x)
             for x in range(nthread)]

    results = P.map(recover_thread, paras)
    res_op = {}
    begin = 0
    middle = 0
    end = 0
    for i in range(nthread):
        thread_num, cur_op, cur_b, cur_m, cur_e = results[i]
        for ele in cur_op:
            if ele not in res_op:
                res_op[ele] = {}
            for k in cur_op[ele]:
                res_op[ele][k] = res_op[ele].get(k, 0) + cur_op[ele][k]
        begin += cur_b
        middle += cur_m
        end += cur_e
    return res_op, begin, middle, end
#
# def rand_string():
#     import random
#     import string
#     length = 50
#     list_str = []
#     P = Pool(40)
#     for i in range(10000):
#         rand_str = ''.join(random.choice(
#                             string.ascii_lowercase
#                             + string.ascii_uppercase
#                             + string.digits)
#                        for i in range(length))
#         list_str.append(rand_str)
#     ground = ''.join(random.choice(
#         string.ascii_lowercase
#         + string.ascii_uppercase
#         + string.digits)
#                      for i in range(length))
#     for i in range(10):
#         print align_all(P, ground, list_str, 40)
#
#
# def main():
#     rand_string()
#
#
# if __name__ == "__main__":
#     main()
