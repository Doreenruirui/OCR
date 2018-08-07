import numpy as np
import sys

file_name = sys.argv[1]
#col = int(sys.argv[2])
#nline = int(sys.argv[3])
#a = np.loadtxt(file_name)[:nline,:]
a = np.loadtxt(file_name + '.ec.txt')
b = np.loadtxt(file_name + '.ew.txt')
print np.mean(a[:,1]/a[:, -1]), np.mean(a[:,0]/a[:, -1]), np.mean(b[:,1]/b[:, -1]), np.mean(b[:,0]/b[:, -1])
#print np.sum(a[:, col])/np.sum(a[:, -1])
