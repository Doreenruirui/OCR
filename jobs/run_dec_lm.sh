source ./bash_function.sh           
folder_name=$1
folder_out=$2
dev=$3
size=$4
machine=$5
jobname=$6
folder_data='/scratch/dong.r/Dataset/OCR/'$folder_name
file_symbol='/scratch/dong.r/Dataset/OCR/voc/ascii.syms'
folder_script='/home/dong.r/OCR/script/lm_dec_10'
file_script=$folder_script'/run.sbatch.'
nline=$(cat $folder_data/'test.x.txt' | wc -l)
nfile=$(ceildiv $nline $size)
echo $nline, $nfile
$(mkdir -p $folder_script)
for i in $(seq 1 $nfile);
do
    cur_file=$file_script$i
    j=$(($i-1))
    start=$(($j * size))
    end=$(($start + $size))
    end=$( (( $nline <= $end)) && echo "$nline" || echo "$end" )
    cur_cmd='python 1_decode_line.py '$folder_name' '$folder_out' '$dev' '$start' '$end' 100 10'
    $(rm_file $cur_file)
    $(writejob $cur_file $jobname $i $folder_script $machine)
    echo 'source ~/.bashrc' >> $cur_file
    echo ''$cur_cmd >> $cur_file
   
    #$(sbatch $cur_file)
done

