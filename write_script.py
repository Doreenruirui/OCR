import numpy as np
from os.path import join as pjoin
import os


folder_script = '/home/dong.r/nlc-master/script/date_dec_train_50_new'
if not os.path.exists(folder_script):
    os.makedirs(folder_script)
num_lines = 246025 
chunk_size = 30000


def write_script(num_lines, chunk_size):
    nsplit = int(np.ceil(num_lines * 1. / chunk_size))
    for i in range(nsplit):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_lines)
        f_out = open(pjoin(folder_script, 'run.sbatch.' + str(start)), 'w')
        f_out.write('#!/bin/bash\n')
        f_out.write('#SBATCH --job-name=%d_dec1_50\n' % (start/10000))
        f_out.write('#SBATCH --output=%s/%d.out\n' % (folder_script, start))
        f_out.write('#SBATCH --error=%s/%d.err\n' % (folder_script, start))
        f_out.write('#SBATCH --exclusive\n')
        f_out.write('#SBATCH --partition=par-gpu-2\n')
        f_out.write('#SBATCH -N 1\n')
        f_out.write('work=/home/dong.r/nlc-master/\n')
        f_out.write('cd $work\n')
        f_out.write('./decode_line_lm_dec.sh %d %d\n' % (start, end))
        f_out.close()

write_script(num_lines, chunk_size)
