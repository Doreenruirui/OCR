from os import listdir
from os.path import join
import gzip
import re
import sys
from multiprocessing import Pool

def process_file(paras):
    file_name, outfile=paras
    with gzip.open(file_name, 'r') as f_:
        content = f_.read()
    content = re.sub('\n', '', content)
    list_doc = re.findall('<DOC .*? >(.*?)</DOC>', content)
    # list_res = []
    with open(outfile, 'w') as f_:
        for doc in list_doc:
            head = re.findall('<HEADLINE>(.*?)</HEADLINE>', doc)
            if len(head) > 0:
                f_.write(head[0] + '\n')
            date = re.findall('<DATELINE>(.*?)</DATELINE>', doc)
            if len(date) > 0:
                f_.write(date[0] + '\n')
            text = re.findall('<TEXT>(.*?)</TEXT>', doc)
            if len(text) > 0:
                paras = re.findall('<P>(.*?)</P>', text[0])
                f_.write('\n'.join(paras) + '\n')
            # cur_doc = [items[0], items[1]]
            # cur_doc += paras
            


def get_file_list(list_fn, out_folder):
    with open(join(out_folder, 'file_list'), 'w') as f_:
        for folder_name in list_fn:
            list_file = listdir(folder_name)
            for fn in list_file:
                if not fn.startswith('.'):
                    out_fn = join(out_folder, fn.replace('.gz', '.txt'))
                    f_.write(join(folder_name, fn) + '\t' + out_fn + '\n')
            #process_file(join(folder_name, fn) , out_fn)

def process_folder(out_folder):
    pool = Pool(20)
    list_file = []
    for line in file(join(out_folder, 'file_list')):
        list_file.append(line.strip('\n').split('\t'))
    pool.map(process_file, list_file)

list_folder = ['/proj/cssh/nulab/corpora/LDC/LDC2011T07/gigaword_eng_5_d1/data/afp_eng', 
                '/proj/cssh/nulab/corpora/LDC/LDC2011T07/gigaword_eng_5_d1/data/apw_eng',
                '/proj/cssh/nulab/corpora/LDC/LDC2011T07/gigaword_eng_5_d2/data/cna_eng',
                '/proj/cssh/nulab/corpora/LDC/LDC2011T07/gigaword_eng_5_d2/data/ltw_eng',
                '/proj/cssh/nulab/corpora/LDC/LDC2011T07/gigaword_eng_5_d2/data/nyt_eng',
                '/proj/cssh/nulab/corpora/LDC/LDC2011T07/gigaword_eng_5_d2/data/wpb_eng',
                '/proj/cssh/nulab/corpora/LDC/LDC2011T07/gigaword_eng_5_d3/data/xin_eng']
get_file_list(list_folder, sys.argv[1])
process_folder(sys.argv[1])
