source /home/dong.r/OCR/jobs/bash_function.sh           
train_id=$1
split_id=$2
error_ratio=100
lm_prob=150
ocr_prob=50
dev=$3
folder_lm=$4
name_script=$5
size=100000
machine=$6
jobname=$7
folder_script='/home/dong.r/OCR/script/'$name_script
file_script=$folder_script'/run.sbatch.'
folder_data='book/'$train_id'/'$split_id
nline=$(cat '/scratch/dong.r/Dataset/OCR/'$folder_data'/'$dev'.x.txt' | wc -l)
nfile=$(ceildiv $nline $size)
echo $nline, $nfile
$(mkdir -p $folder_script)
for i in $(seq 1 $nfile);
do
    cur_file=$file_script$i
    j=$(($i-1))
    cur_start=$(($j*$size))
    cur_end=$(($i * $size))
    cur_cmd='python data_noisy_book.py '$train_id' '$split_id' '$error_ratio' '$lm_prob' '$ocr_prob' '$dev' '$folder_lm' '$cur_start' '$cur_end
    $(rm_file $cur_file)
    $(writejob $cur_file $jobname $i $folder_script $machine)
    echo ''$cur_cmd >> $cur_file
    #$(sbatch $cur_file)
done
