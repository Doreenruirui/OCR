import numpy as np
import sys
file_name = sys.argv[1]
col = int(sys.argv[2])
a = np.loadtxt(file_name)
print np.mean(a[:,col]/a[:,-1])
print np.sum(a[:,col])/np.sum(a[:,-1])
