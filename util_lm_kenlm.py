import kenlm
from os.path import join as pjoin
import re


def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def initialize(folder_lm):
    global lm
    lm = kenlm.LanguageModel(pjoin(folder_lm, 'train.arpa'))


def get_string_to_score(sent):
    sent = remove_nonascii(sent)
    items = []
    for ele in sent:
        if len(ele.strip()) == 0:
            items.append('<space>')
        else:
            items.append(ele)
    return ' '.join(items)


def score_sent(paras):
    global lm
    thread_no, sent = paras
    sent = get_string_to_score(sent.lower())
    return thread_no, lm.score(sent)



