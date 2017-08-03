source ./bash_function.sh           
merge_file(){
            nmerge=$1;
            bashname=$4;
            filename=$3;
            nfile=$2
            for i in $(seq 1 $nmerge);
            do
                 end=$(($i * 2))
                 start=$(($end - 1))
                 end=$( (( $nfile <= $end)) && echo "$nfile" || echo "$end" )
                 file1=$filename$start
                 file2=$filename$end
                 #echo $start >> $bashname
                 #echo $end >> $bashname
                 if [ $start -ne $end ];
                 then
                    echo 'ngrammerge '$file1' '$file2' > '$filename$i'.new &' >> $bashname
                 fi
             done
             echo wait >> $bashname
             for i in $(seq 1 $nmerge);
             do
                  end=$(($i * 2))
                  start=$(($end - 1))
                  end=$( (( $nfile <= $end)) && echo "$nfile" || echo "$end" )
                  file1=$filename$start
                  file2=$filename$end
                  if [ $start -ne $end ];  
                  then
                    echo 'rm '$file1 >> $bashname
                    echo 'rm '$file2 >> $bashname
                  else
                    echo 'mv '$file1' '$filename$i'.new' >> $bashname
                  fi
             done
          }
folder_name=$1
size=$2
machine=$3
jobname=$4
folder_data='/scratch/dong.r/Dataset/OCR/'$folder_name
file_symbol='/scratch/dong.r/Dataset/OCR/voc/ascii.syms'
folder_script='/home/dong.r/OCR/script/lm_gutern'
file_script=$folder_script'/run.sbatch.'
nline=$(cat $folder_data/train.text | wc -l)
nfile=$(ceildiv $nline $2)
echo $nline, $nfile
$(mkdir -p $folder_script)
for i in $(seq 1 $nfile);
do
    cur_file=$file_script$i
    j=$(($i-1))
    k=$(($j*$2))
    cur_cmd='./run_lm_char_line.sh '$1' '$i' '$k' '$2' '$nline
    $(rm_file $cur_file)
    $(writejob $cur_file $jobname $i $folder_script $machine)
    echo ''$cur_cmd >> $cur_file
    $(sbatch $cur_file)
done
merge_file=$file_script'merge'
$(rm_file $merge_file)
niter=$(log2 $nfile)
nmerge=$nfile
$(writejob $merge_file $jobname 'merge' $folder_script $machine)
for i in $(seq 1 $niter);
do
    echo echo $i >> $merge_file
    echo $nmerge
    last=$nmerge
    nmerge=$(ceildiv $nmerge 2)
    $(merge_file $nmerge $last $folder_data'/train_symbols.cnt.' $merge_file) 
    for j in $(seq 1 $nmerge);
    do
        echo 'mv '$folder_data'/train_symbols.cnt.'$j'.new ' $folder_data'/train_symbols.cnt.'$j >> $merge_file
    done
done
echo 'mv '$folder_data'/train_symbols.cnt.1 '$folder_data'/train_symbols.cnt' >> $merge_file
echo 'ngrammake '$folder_data'/train_symbols.cnt >'$folder_data'/train.mod' >> $merge_file

#para=$(echo $para | xargs)
#para=($para)
#./run_lm_char_line.sh $1 ${para[@]} $2 $nline

#echo ${#para}
#echo $para
#echo $all_file
#$($all_file)
#farcompilestrings -token_type=utf8 -keep_symbols=1 $folder_data/train.text >$folder_data/train.far
#ngramcount -order=5 --require_symbols=false $folder_data/train.far >$folder_data/train.cnt
#fstsymbols --isymbols=$file_symbol --osymbols=$file_symbol  $folder_data/train.cnt $folder_data/train_with_symbols.cnt
#ngrammake $folder_data/train_with_symbols.cnt >$folder_data/train.mod
