source ./bash_function.sh
cur_folder=$1
cur_prefix=$2
cur_out=$3
beam=$4
machine=$5
jobname=$6
lm_name=$7
folder_script=$8
folder_data='/scratch/dong.r/Dataset/OCR/'
folder_lm=$folder_data'lm/char/'$lm_name
folder_script='/home/dong.r/OCR/script/'$folder_script
file_script=$folder_script'/run.sbatch.'
nline=$(cat $folder_data'/'$cur_folder'/'$cur_prefix'.x.txt' | wc -l)
chunk_size=10000
nchunk=$(ceildiv $nline $chunk_size)
echo $nline, $nchunk
$(mkdir -p $folder_script)
weight1=(100, 1000)
weight2=(2 4 6 8 10 12 14 16 18)
for i in $(seq 1 2);
do
    for w1 in 100.0 1000.0;
    do
        for w2 in 2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0;
            do
                cur_file=$file_script$i'.'$w1'_'$w2'_'$lm_name
                $(rm_file $cur_file)
                j=$(($i - 1))
                start=$(($j * $chunk_size))
                end=$(($i * chunk_size))
                end=$( (($nline <= $end)) && echo "$nline" || echo "$end" )
                $(writejob $cur_file $jobname $i'.'$w1'.'$w2'.'$lm_name $folder_script $machine)
                echo python decode_line.py $cur_folder $cur_out $cur_prefix $start $end $beam $w1 $w2 $folder_lm>> $cur_file
                #$(sbatch $cur_file)
            done
    done
done
