from os.path import join as pjoin

folder_data = '/scratch/dong.r/Dataset/OCR/lm/char'
#folder_lm = 'richmond_0_0_low'
folder_lm = 'nyt_low'


def prepare_data():
     with open(pjoin(folder_data, folder_lm, 'train.text.char'), 'w') as f_:
        for line in file(pjoin(folder_data, folder_lm, 'train.text')):
            line = line.strip().lower()
            items = []
            for ele in line:
                if len(ele.strip()) == 0:
                    items.append('<space>')
                else:
                    items.append(ele)
            f_.write(' '.join(items) + '\n')

prepare_data()
