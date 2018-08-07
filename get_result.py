from os.path import join as pjoin
import sys
import numpy as np

folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR'
arg_index = sys.argv[1]
arg_data = sys.argv[2]

index = np.loadtxt(pjoin(folder_data, arg_index), dtype=int)
lines = np.loadtxt(pjoin(folder_data, arg_data), dtype=int)
np.savetxt(pjoin(folder_data, arg_data), lines[index,:], fmt='%d')
