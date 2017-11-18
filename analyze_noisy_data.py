from util_lm_kenlm import get_string_to_score
import kenlm
from os.path import join as pjoin

lm = None
folder_data = '/scratch/dong.r/Dataset/OCR/book1800/single/0/0/train'
def mean_y_score():
    global lm
    lm = kenlm.LanguageModel('/scratch/dong.r/Dataset/OCR/lm/char/nyt_low/train.arpa')
    lmscore = []
    lmlen = []
    for line in file(pjoin(folder_data, 'dev.y.txt')):
        #items = [ele.strip().lower() for ele in line.strip('\n').split('\t')]
        #lmscore.append([lm.score(get_string_to_score(ele)) for ele in items])
        #lmlen.append([len(ele) for ele in items])
        lmscore.append([lm.score(get_string_to_score(line.strip().lower())), len(line.strip())])
    with open(pjoin(folder_data, 'dev.y.score.txt'), 'w') as f_:
        for ele in lmscore:
            f_.write(str(ele[0]) + '\t' + str(ele[1]) + '\n')
        #for score, lens in zip(lmscore, lmlen):
        #    f_.write('\t'.join(map(str, score)) + ';' + '\t'.join(map(str, lens)) + '\n')
           
mean_y_score()
