source /home/dong.r/OCR/jobs/bash_function.sh           
folder_data=$1
file_model=$2
folder_tmp=$3
folder_out=$4
name_script=$5
dev=$6
size=$7
machine=$8
jobname=$9
folder_root='/scratch/dong.r/Dataset/OCR/'
model_root='/scratch/dong.r/Model/OCR/pcrf/'
folder_script='/home/dong.r/OCR/script/'$name_script
file_script=$folder_script'/run.sbatch.'
nline=$(cat $folder_root$folder_data'/'$dev | wc -l)
nfile=$(ceildiv $nline $size)
echo $nline, $nfile
$(mkdir -p $folder_script)
for i in $(seq 1 $nfile);
do
    cur_file=$file_script$i
    j=$(($i-1))
    cur_start=$(($j*$size))
    cur_end=$(($i * $size))
    true_start=$(($cur_start + 1))
    cmd1='sed -n "'$true_start', '$cur_end' p" '$folder_root$folder_data'/'$dev' > '$folder_root$folder_data'/'$dev'.'$cur_start'_'$cur_end 
    #echo $folder_root$folder_data'/'$folder_out
    #echo $model_root$folder_out'/'$dev'.out'
    cur_cmd='./test_dev.sh '$folder_root$folder_data'/'$dev'.'$cur_start'_'$cur_end' '$model_root$file_model' 6 '$folder_tmp' '$folder_root$folder_data'/'$folder_out' > '$folder_root$folder_data'/'$folder_out'/'$dev'.out.'$cur_start'_'$cur_end
    $(rm_file $cur_file)
    $(writejob $cur_file $jobname $i $folder_script $machine)
    echo ''$cmd1 >> $cur_file
    echo ''$cur_cmd >> $cur_file
    #$(sbatch $cur_file)
done
