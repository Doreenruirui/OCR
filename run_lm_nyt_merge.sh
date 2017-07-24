#!/usr/bin/env bash
source ./bash_function.sh
merge_file(){
            local start=$1
            local end=$2
            local filename=$3
            local bashname=$4
            local nmerge=$5
            local cur_start=0
            local cur_end=0
            for i in $(seq 1 $nmerge);
            do
                 cur_end=$(($i * 2))
                 cur_start=$(($start + $cur_end - 2))
                 cur_end=$(($start + $cur_end - 1))
                 cur_end=$( (($end <= $cur_end)) && echo "$end" || echo "$cur_end" )
                 file1=$filename$cur_start
                 file2=$filename$cur_end
                 #echo $start >> $bashname
                 #echo $end >> $bashname
                 cur_no=$(($i + $start - 1))
                 if [ $cur_start -ne $cur_end ];
                 then
                    echo 'ngrammerge '$file1' '$file2' > '$filename$cur_no'.new &' >> $bashname
                 fi
             done
             echo wait >> $bashname
             for i in $(seq 1 $nmerge);
             do
                  cur_end=$(($i * 2))
                  cur_start=$(($start + $cur_end - 2))
                  cur_end=$(($start + $cur_end - 1))
                  cur_end=$( (($end <= $cur_end)) && echo "$end" || echo "$cur_end" )
                  file1=$filename$cur_start
                  file2=$filename$cur_end
                  cur_no=$(($i + $start - 1))
                  if [ $cur_start -ne $cur_end ];
                  then
                    echo 'rm '$file1' &' >> $bashname
                    echo 'rm '$file2' &' >> $bashname
                  else
                    echo 'mv '$file1' '$filename$cur_no'.new &' >> $bashname
                  fi
                  
             done
             echo 'wait' >> $bashname
          }

merge_chunk(){
    local start=$1
    local end=$2
    local merge_file=$3'.'$end
    local nfile=$(($end + 1 - $start))
    local niter=$(log2 $nfile)
    local nmerge=$nfile
    $(writejob $merge_file $jobname$start 'merge.'$end $folder_script $machine)
    for i in $(seq 1 $niter);
    do
        echo $nmerge
        cur_end=$(($start + $nmerge - 1))
        nmerge=$(ceildiv $nmerge 2)
        $(merge_file $start $cur_end $folder_data'/train_symbols.cnt.' $merge_file $nmerge)
        for j in $(seq 1 $nmerge);
        do
            cur_no=$(($j + $start - 1))
            echo 'mv '$folder_data'/train_symbols.cnt.'$cur_no'.new ' $folder_data'/train_symbols.cnt.'$cur_no' &' >> $merge_file
        done
        echo 'wait' >> $merge_file
    done
}


folder_name=$1
machine=$2
jobname=$3
chunk_size=$4
folder_input='/scratch/dong.r/Dataset/unprocessed/NYT'
folder_data='/scratch/dong.r/Dataset/OCR/'$folder_name
file_symbol='/scratch/dong.r/Dataset/OCR/voc/ascii.syms'
folder_script='/home/dong.r/nlc-master/script/lm_nyt'
file_script=$folder_script'/run.sbatch.'
nfile=$(cat $folder_input'/out_file' | wc -l)
echo $nfile
merge_file=$file_script'merge'
nchunk=$(ceildiv $nfile $chunk_size)
$(rm_file $merge_file)
for i in $(seq 1 $nchunk);
do
    j=$(($i -1))
    cur_start=$(($j * $chunk_size + 1))
    cur_end=$(($cur_start + $chunk_size - 1))
    cur_end=$( (($nfile <= $cur_end)) && echo "$nfile" || echo "$cur_end" )
    $(merge_chunk $cur_start $cur_end $merge_file)
    echo mv $folder_data/train_symbols.cnt.$cur_start $folder_data/train_symbols.cnt.$i >> $merge_file'.'$nchunk
done
$(merge_chunk 1 $nchunk $merge_file)
echo 'mv '$folder_data'/train_symbols.cnt.1 '$folder_data'/train_symbols.cnt' >> $merge_file'.'$nchunk
echo 'ngrammake --backoff=true '$folder_data'/train_symbols.cnt >'$folder_data'/train.mod' >> $merge_file'.1'$nchunk
