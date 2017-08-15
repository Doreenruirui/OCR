source ./bash_function.sh           
folder_name=$1
machine=$2
jobname=$3
folder_input='/scratch/dong.r/Dataset/unprocessed/NYT'
folder_data='/scratch/dong.r/Dataset/OCR/'$folder_name
file_voc='/scratch/dong.r/Dataset/OCR/voc/ascii.syms'
folder_script='/home/dong.r/OCR/script/lm_nyt_upper'
file_script=$folder_script'/run.sbatch.'
nfile=$(cat $folder_input'/out_file_non_empty' | wc -l)
echo $nfile
$(mkdir -p $folder_script)
for i in $(seq 1 $nfile);
do
    echo $i
    filename=$(sed -n ''$i,$i'p' $folder_input'/out_file_non_empty')
    cur_file=$file_script$i
    #cur_cmd='./run_lm_char_line.sh '$folder_input' '$filename $i
    $(rm_file $cur_file)
    $(writejob $cur_file $jobname $i $folder_script $machine)
    echo 'source ~/.bashrc' >> $cur_file
    echo 'farcompilestrings -token_type=utf8 -keep_symbols=1 '$filename' > '$folder_data'/train.far.'$i >> $cur_file
    echo 'ngramcount -order=5  --require_symbols=false '$folder_data'/train.far.'$i' > '$folder_data'/train.cnt.'$i >> $cur_file
    echo 'fstsymbols --isymbols='$file_voc' --osymbols='$file_voc' '$folder_data'/train.cnt.'$i' > '$folder_data'/train_symbols.cnt.'$i >> $cur_file
    echo 'rm '$folder_data'/train.far.'$i >> $cur_file
    echo 'rm '$folder_data'/train.cnt.'$i >> $cur_file
    #echo ''$cur_cmd >> $cur_file
    #$(sbatch $cur_file)
done
