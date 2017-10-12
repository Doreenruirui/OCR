from levenshtein import count_pair
from multiprocessing import Pool


list_x = []
list_y = []
for line in ('/scratch/dong.r/Dataset/OCR/book/0/0/50/test_20/20/100_17/man_wit.test.avg.bs.txt'):
    list_x.append(line.strip('\n'))
for line in ('/scratch/dong.r/Dataset/OCR/book/0/0/50/test_20/20/100_17/man_wit.test.bs.txt'):
    list_y.append(line.strip('\n'))
pool = Pool(100)
i1, d1, r1 = count_pair(pool, list_x, list_y)
print i1, d1, r1
