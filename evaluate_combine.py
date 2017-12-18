import numpy as np
import sys
from os.path import join

folder_data = sys.argv[1]
a = [] 
for line in file(join(folder_data, 'man_wit.test.ew.txt')):
    items = map(float, line.strip().split())
    a.append(items)
 
#for line in file(join(folder_data, 'man.test.ec.txt')):
#    items = map(float, line.strip().split())
#    a.append(items)

b = [] 
for ele in a:
    if len(ele) == 2:
        b.append([ele[0], ele[0], ele[1]])
    else:
        b.append([min(ele[:-1]), ele[0], ele[-1]])
b = np.asarray(b)
print np.mean(b[:,1] / b[:,-1]), np.sum(b[:,1]) / np.sum(b[:,-1]), np.mean(b[:,0]/b[:,-1]), np.sum(b[:,0])/np.sum(b[:,-1])

