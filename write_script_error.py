import numpy as np
from os.path import join as pjoin
import os


folder_script = '/home/dong.r/nlc-master/script/process_data/other'
if not os.path.exists(folder_script):
    os.makedirs(folder_script)
num_lines = 134977312
chunk_size = 1000000


def write_script(num_lines, chunk_size):
    nsplit = int(np.ceil(num_lines * 1. / chunk_size))
    f_run = open(pjoin(folder_script, 'run_all.sh'), 'w')
    for i in range(nsplit):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_lines)
        f_out = open(pjoin(folder_script, 'run.sbatch.' + str(start/1000000)), 'w')
        f_out.write('#!/bin/bash\n')
        f_out.write('#SBATCH --job-name=%d\n' % (start/1000000))
        f_out.write('#SBATCH --output=%s/%d.out\n' % (folder_script, start/1000000))
        f_out.write('#SBATCH --error=%s/%d.err\n' % (folder_script, start/1000000))
        f_out.write('#SBATCH --exclusive\n')
        f_out.write('#SBATCH --partition=ser-par-10g-2\n')
        f_out.write('#SBATCH -N 1\n')
        f_out.write('work=/home/dong.r/nlc-master/\n')
        f_out.write('cd $work\n')
        f_out.write('python process_out_of_domain.py error o %d %d\n' % (start, end))
        f_out.close()
        f_run.write('sbatch run.sbatch.' + str(start/1000000) +'\n')
    f_run.close()
    
write_script(num_lines, chunk_size)
