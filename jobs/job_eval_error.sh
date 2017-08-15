#!/usr/bin/env bash
source /home/dong.r/OCR/jobs/bash_function.sh
cur_folder=$1
cur_out=$2
cur_prefix=$3
beam=$4
size=$5
machine=$6
jobname=$7
script_name=$8
folder_data='/scratch/dong.r/Dataset/OCR/'
folder_script='/home/dong.r/OCR/script/'$script_name
file_script=$folder_script'/run.sbatch.'
echo $file_script
nline=$(cat $folder_data'/'$cur_folder'/'$cur_prefix'.x.txt' | wc -l)
chunk_size=$size
nchunk=$(ceildiv $nline $chunk_size)
echo $nline, $nchunk
$(mkdir -p $folder_script)
for i in $(seq 1 $nchunk);
do
    cur_file=$file_script$i
    $(rm_file $cur_file)
    j=$(($i - 1))
    start=$(($j * $chunk_size))
    end=$(($i * chunk_size))
    #end=$( (($nline <= $end)) && echo "$nline" || echo "$end" )
    $(writejob $cur_file $jobname $i $folder_script $machine)
    echo python evaluate_error_rate_multi.py $cur_folder $cur_out $cur_prefix $beam $start $end >> $cur_file
done
