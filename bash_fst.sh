source ./bash_function.sh

write_fst(){
    local line=$1
    local file_fst=$2
    string=$(echo -e "$line" | cut -d $'\t' -f1)
    weight=$(echo -e "$line" | cut -d $'\t' -f2)
    $(rm_file $bf)
    local len=${#string}
    #len=$(($len -1))
    for i in $(seq 1 $len);
    do
        j=$(($i - 1))
        char=${string:$j:1}
        char=$( [[ $char == [[:space:]] ]] && echo '<space>' || echo $char )
        j=$(($i - 1))
        cur_w=$( (( $i == 1)) && echo $weight || echo '0' )
        cur_arc=$j'\t'$i'\t'$char'\t'$char'\t'$cur_w
        echo -e $cur_arc >> $file_fst
    done
    echo -e $len'\t'0 >> $file_fst
    }


write_fst_batch(){
    local string=$1
    local line_no=$2
    local file_no=$3
    local file_fst=$4
    local file_bf=$5
    IFS=$'\n' y=($string)
    nline=${#string}
    for i in $(seq 1 $nline);
    do
        cur_str="${y[i]}"
        cur_fst_file=$file_fst$file_no'.'$line_no'.'$i
        echo 'write_fst '$cur_str' '$cur_fst_file' &' >> $file_bf
    done
    echo wait >> $file_bf
    for i in $(seq 1 $nline);
    do
        cur_fst_file=$file_fst$file_no'_'$line_no'.'$i
        echo 'fstcompile --isymbols='$symbol_file' --osymbols='$symbol_file' --keep_isymbols --keep_osymbols '$cur_fst_file' '$cur_fst_file' & ' >> $file_bf
    done
    echo wait >> $file_bf
    }

merge_fst(){
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
            echo 'fstunion '$file1' '$file2' | fstrmepsilon - | fstdeterminize - | fstminimize - > '$filename$i'.new &' >> $bashname
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

concat_fst(){
    nmerge=$1;
    bashname=$4;
    filename=$3;
    nfile=$2
    file_inter=$5
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
            echo 'fstconcat '$file1' '$file_inter' | fstconcat - '$file2' > '$filename$i'.new &' >> $bashname
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


merge_iter(){
    local nfile=$1
    task=$2
    file_bash=$3
    file_name=$4
    niter=$(log2 $nfile)
    nmerge=$nfile
    for i in $(seq 1 $niter);
    do
        echo echo $i >> $merge_file
        echo $nmerge
        last=$nmerge
        nmerge=$(ceildiv $nmerge 2)
        $($task $nmerge $last $file_name $file_bash)
         for j in $(seq 1 $nmerge);
        do
            echo 'mv '$filename$i'.new '$filename$i >> $file_bash
        done
    done
}


get_fst_for_line(){
    line_no=$1
    file_fst=$2
    file_name=$3
    file_bf=$4
    file_no=$(echo -e "$line" | cut -d $'_' -f1 | cut -d $'.' -f4)
    start=$(($(($(($line_no - 1)) * 100)) + 1))
    end=$(($line_no * 100))
    nline=$100
    a=$(sed -n ''$start','$end'p' $file_name)
    $(write_fst_batch "$a" $line_no $file_no $file_fst $file_bf)
    $(merge_iter $nline merge_fst $file_bf $file_fst$file_no'.'$line_no'.')
}


get_fst_for_group(){
    start=$1
    end=$2
    file_name=$3
    file_fst=$4
    file_no=$(echo -e "$line" | cut -d $'_' -f1 | cut -d $'.' -f4)
    file_bf='run_group.'$file_no'.'$start
    for i in $(seq $start $end);
    do
        $(get_fst_for_line $i $file_fst $file_name $file_bf'.'$i)
    done
    for i in $(seq $start $end);
    do
        echo 'chmod a+x '$file_bf'.'$i' | ./'$file_bf'.'$i' | rm '$file_bf'.'$i' &'>> $file_bf
    done

}

folder_data='/scratch/dong.r/Dataset/OCR/'$1
file_symbol='/scratch/dong.r/Dataset/OCR/voc/ascii.syms'
file_lm='/scratch/dong.r/Dataset/OCR/lm/char/richmond/train.mod'
file_fst=$folder_data'/fst.tmp.'

#$(write_fst $"one two three" 1.fst)
