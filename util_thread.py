import numpy as np

def callMethod(o, name):
    getattr(o, name)()

def task_per_thread(ntask, nthread):
    return int(np.floor(ntask * 1. / nthread))

# def split_task(pool, nthread, task, list_paras, share_paras):
#     ntask = len(list_paras[0])
#     ncand = task_per_thread(ntask, nthread)
#     list_input = []
#     ninput = len(list_paras)
#     for x in range(nthread):
#         cur_input = (x, )
#         cur_input += tuple(share_paras)
#         for i in range(ninput):
#             cur_input += (list_paras[i][x * ncand : min((x + 1) * ncand, ntask)], )
#         list_input.append(cur_input)
#     results = pool.map(task, list_input)
#     res = [None for i in range(ntask)]
#     for i in range(nthread):
#         thread_num, cur_res = results[i]
#         res[thread_num] = cur_res
#     if None in res:
#         raise 'Something Wrong!'
#     return res


def split_task(pool, task, list_paras):
    list_input = []
    ntask = len(list_paras[0])
    ninput = len(list_paras)
    for x in range(ntask):
        cur_input = (x, )
        for i in range(ninput):
            cur_input += (list_paras[i][x], )
        list_input.append(cur_input)
    results = pool.map(task, list_input)
    res = [None for i in range(ntask)]
    for i in range(ntask):
        thread_num, cur_res = results[i]
        res[thread_num] = cur_res
    if None in res:
        raise 'Something Wrong!'
    return res