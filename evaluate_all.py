import numpy as np
import sys

folder_data = sys.argv[1]
for fn in ['single.test', 'multi.test']:
    for fc in ['ec', 'ew']:
        a = np.loadtxt(folder_data + '/' + fn + '.' + fc + '.txt')
        print np.mean(a[:,0] / a[:,-1])
        print np.sum(a[:,0])/np.sum(a[:,-1])
        print np.mean(a[:,1]/ a[:,-1])
        print np.sum(a[:,1]) / np.sum(a[:,-1])

