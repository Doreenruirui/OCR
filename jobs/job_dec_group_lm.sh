source ./bash_function.sh           
folder_name=$1
folder_lm=$2
folder_out=$3
name_script=$4
dev=$5
size=$6
machine=$7
jobname=$8
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
    cur_cmd='python decode_line_multi_lm.py /scratch/dong.r/Dataset/OCR '$folder_name' '$folder_out' '$folder_lm' '$dev' '$cur_start' '$cur_end
    $(rm_file $cur_file)
    $(writejob $cur_file $jobname $i $folder_script $machine)
    echo ''$cur_cmd >> $cur_file
    #$(sbatch $cur_file)
done
