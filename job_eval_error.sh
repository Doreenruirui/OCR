source ./bash_function.sh
cur_folder=$1
cur_prefix=$2
cur_out=$3
beam=$4
machine=$5
jobname=$6
folder_data='/scratch/dong.r/Dataset/OCR/'
folder_script='/home/dong.r/OCR/script/dec_eval'
file_script=$folder_script'/run.sbatch.'
nline=$(cat $folder_data'/'$cur_folder'/'$cur_prefix'.x.txt' | wc -l)
chunk_size=10000
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
    end=$( (($nline <= $end)) && echo "$nline" || echo "$end" )
    $(writejob $cur_file $jobname $i $folder_script $machine)
    echo python evaluate_error_rate.py $cur_folder $cur_prefix $cur_out $beam $start $end >> $cur_file
done
