from os.path import join as pjoin


DEV_PATH = '/home/rui/Dataset/OCR/data/char/dev.'


def merge():
    reverse_path = DEV_PATH + 'or.txt'
    output_path = DEV_PATH + 'o.txt'
    input_path = DEV_PATH + 'x.txt'
    with open(input_path, 'r') as f_:
        inputs = f_.readlines()
    with open(reverse_path, 'r') as f_:
        reverse = f_.readlines()
        reverse = reverse[::-1]
    with open(output_path, 'r') as f_:
        outputs = f_.readlines()
    num_reverse = len(inputs) - len(outputs)
    with open(DEV_PATH + 'p.txt', 'a') as f_:
        for line in outputs:
            f_.write(line)
        for line in reverse[-num_reverse:]:
            f_.write(line)
    print len(inputs), len(outputs), num_reverse

merge()