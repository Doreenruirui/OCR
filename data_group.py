from os.path import join, exists
import os

folder_multi = '/scratch/dong.r/Dataset/OCR/multi'


def get_train_data(train_id, split_id, error_ratio, lm_prob, ocr_prob, train, lm_name):
    folder_train = join(folder_multi, str(train_id), str(split_id))
    list_info = []
    for line in file(join(folder_train, train + '.y.txt')):
        list_info.append(line)
    folder_out = join(folder_train, 'noisy_' + str(error_ratio) + '_' + str(lm_prob) + '_' + str(ocr_prob))
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    f_x = open(join(folder_out, train + '.x.txt'), 'w')
    f_y = open(join(folder_out, train + '.y.txt'), 'w')
    f_z = open(join(folder_out, train + '.z.txt'), 'w')
    for i in range(len()):
        print i
        cur_x = [ele.strip() for ele in list_x[i].strip('\n').split('\t') if len(ele.strip()) > 0]
        best_str, best_id, best_prob, probs = rank_sent(pool, cur_x)
        if - best_prob / len(cur_x[best_id]) <= lm_prob * 0.01:
            if best_prob / probs[0] <= error_ratio * 0.01 and - probs[0] / len(cur_x[0]) < 0.01 * ocr_prob:
                f_x.write(cur_x[0] + '\n')
                f_y.write(best_str + '\n')
                f_z.write(list_y[i])
    f_x.close()
    f_y.close()
    f_z.close()