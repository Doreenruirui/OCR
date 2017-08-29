import numpy as np
# from multiprocessing import Pool

output = None
output_str = None


def align(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
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
    d = np.ones((len1 + 1, len2 + 1), dtype=int) * 1000000
    op = np.zeros((len1 + 1, len2 + 1), dtype=int)
    if len1 == 0:
        return len2, d * 0, op
    if len2 == 0:
        return len1, d * 0, op
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


def count_operation(paras):
    str1, str2 = paras
    if len(str1) == 0:
        return len(str2), 0, 0
    elif len(str2) == 0:
        return 0, len(str1), 0
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
    num_ins = 0
    num_del = 0
    num_rep = 0
    for (i, j) in path:
        if op[i, j] == 1:
            num_ins += 1
        elif op[i, j] == 2:
            num_del += 1
        elif op[i, j] == 3:
            num_rep += 1
    return num_ins, num_del, num_rep


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
    thread_num, str1, list_str, flag_char = para
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


def align_one2one(para):
    thread_num, str1, str2, flag_char = para
    if flag_char:
        return thread_num, align(str1, str2)
    else:
        return thread_num, align(str1.split(), str2.split())


def recover_thread(para):
    thread_num, str1, str2 = para
    res_op = {}
    cur_op, cur_b, cur_m, cur_e = recover(str1.strip(), str2.strip())
    for ele in cur_op:
        if ele not in res_op:
            res_op[ele] = {}
        for k in cur_op[ele]:
            res_op[ele][k] = res_op[ele].get(k, 0) + cur_op[ele][k]
    return thread_num, res_op, cur_b, cur_m, cur_e


def recover_pair(P, truth, cands, nthread):
    global output, output_str
    output = np.zeros(nthread)
    output_str = ['' for _ in range(nthread)]
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


def align_one2many(P, truth, cands, flag_char=1):
    global output, output_str
    ncand = len(cands)
    list_truth = [truth for _ in range(ncand)]
    list_index = np.arange(ncand).tolist()
    list_flag = [flag_char for _ in range(ncand)]
    paras = zip(list_index, list_truth, cands, list_flag)
    results = P.map(align_one2one, paras)
    min_dis = float('inf')
    min_str = ''
    for i in range(ncand):
        cur_thread, cur_dis = results[i]
        if cur_dis < min_dis:
            min_dis = cur_dis
            min_str = cands[cur_thread]
    return min_dis, min_str


def align_pair(P, truth, cands, flag_char=1):
    global output, output_str
    ndata = len(truth)
    output = [0 for _ in range(ndata)]
    list_index = np.arange(ndata).tolist()
    list_flag = [flag_char for _ in range(ndata)]
    paras = zip(list_index, truth, cands, list_flag)
    results = P.map(align_one2one, paras)
    for i in range(ndata):
        thread_num, dis = results[i]
        output[thread_num] = dis
    return output


def count_pair(P, truth, cands):
    global output, output_str
    ndata = len(truth)
    paras = zip(truth, cands)
    # results = []
    # for ele in paras:
    #     results.append(count_operation(ele))
    results = P.map(count_operation, paras)
    num_ins, num_del, num_rep = 0, 0, 0
    for i in range(ndata):
        cur_ins, cur_del, cur_rep = results[i]
        num_ins += cur_ins
        num_del += cur_del
        num_rep += cur_rep
    return num_ins, num_del, num_rep


def align_beam(P, truth, cands, flag_char=1):
    global output, output_str
    ndata = len(truth)
    list_dis = [0 for _ in range(ndata)]
    list_str = [[] for _ in range(ndata)]
    list_index = np.arange(ndata).tolist()
    list_flag = [flag_char for _ in range(ndata)]
    paras = zip(list_index, truth, cands, list_flag)
    results = P.map(align_one2many_thread, paras)
    for i in range(ndata):
        cur_thread, cur_dis, cur_str = results[i]
        list_dis[cur_thread] = cur_dis
        list_str[cur_thread] = cur_str
    return list_dis, list_str
