source /home/dong.r/OCR/jobs/bash_function.sh           
folder_name=$1
folder_out=$2
name_script=$3
dev=$4
size=$5
machine=$6
jobname=$7
folder_script='/home/dong.r/OCR/script/'$name_script
file_script=$folder_script'/run.sbatch.'
nline=$(cat '/scratch/dong.r/Dataset/OCR/'$folder_name'/'$dev'.x.txt' | wc -l)
nfile=$(ceildiv $nline $size)
echo $nline, $nfile
$(mkdir -p $folder_script)
for i in $(seq 1 $nfile);
do
    cur_file=$file_script$i
    j=$(($i-1))
    cur_start=$(($j*$size))
    cur_end=$(($i * $size))
    cur_cmd='python decode_line_rmb_2.py '$folder_name' '$folder_out' '$dev' '$cur_start' '$cur_end
    $(rm_file $cur_file)
    $(writejob $cur_file $jobname $i $folder_script $machine)
    echo ''$cur_cmd >> $cur_file
    #$(sbatch $cur_file)
done
