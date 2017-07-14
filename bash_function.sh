ceildiv(){ echo $((($1+$2-1)/$2)); }

writejob(){ echo '#!/bin/bash' >> $1;
            echo '#SBATCH --job-name='$2'.'$3 >> $1;
            echo '#SBATCH --output='$4'/out.'$3 >> $1;
            echo '#SBATCH --error='$4'/err.'$3 >> $1;
            echo '#SBATCH --exclusive' >> $1;
            echo '#SBATCH --partition='$5 >> $1;
            echo '#SBATCH -N 1' >> $1;
            echo '' >> $1;
            echo 'work=/home/dong.r/nlc-master/' >> $1;
            echo 'cd $work' >> $1;
            }


log2() {
        local x=0
        for (( y=$1-1 ; $y > 0; y >>= 1 )) ; do
        let x=$x+1
        done
        echo $x
        }

rm_file(){
        if [ -f $1 ]
        then
            rm $1
        fi
      }




merge_fst(){
    local nmerge=$1
    local nfile=$2
    local filename=$3
	local bashname=$4
    local task_no=$5
	local start=0
	local end=0
	local file1=''
	local file2=''
    case $task_no in
    1)
        task="fstunion @ @ | fstrmepsilon - | fstdeterminize - | fstminimize - "
        ;;
    2)
        task='fstconcat @ /scratch/dong.r/Dataset/OCR/lm/concat.fst | fstconcat - @ '
        ;;
    esac
    #echo $task_no >> $bashname
    local task1=$(echo -e "$task" | cut -d $'@' -f1)
    local task2=$(echo -e "$task" | cut -d $'@' -f2)
    local task3=$(echo -e "$task" | cut -d $'@' -f3)
#	echo $task1, $task2, $task3
    for i in $(seq 1 $nmerge);
    do
         end=$(($i * 2))
         start=$(($end - 1))
         end=$( (( $nfile <= $end)) && echo "$nfile" || echo "$end" )
         file1=$filename'.'$start
         file2=$filename'.'$end
         if [ $start -ne $end ];
         then
			echo $task1$file1$task2$file2$task3'> '$filename'.'$i'.new &' >> $bashname
            #echo 'fstconcat '$file1' '$file_inter' | fstconcat - '$file2' > '$filename$i'.new &' >> $bashname
            #echo 'fstunion '$file1' '$file2' | fstrmepsilon - | fstdeterminize - | fstminimize - > '$filename$i'.new &' >> $bashname
         fi
     done
     echo wait >> $bashname
     for i in $(seq 1 $nmerge);
     do
          end=$(($i * 2))
          start=$(($end - 1))
          end=$( (( $nfile <= $end)) && echo "$nfile" || echo "$end" )
          file1=$filename'.'$start
          file2=$filename'.'$end
          if [ $start -ne $end ];
          then
            echo 'rm '$file1' &'>> $bashname
            echo 'rm '$file2' &' >> $bashname
          else
            echo 'mv '$file1' '$filename'.'$i'.new &' >> $bashname
          fi
     done
     echo wait >> $bashname
    }


merge_iter(){
    local nfile=$1
	local task=$2
    local file_bash=$3
    local file_name=$4
	local niter=$(log2 $nfile)
    local nmerge=$nfile
    local last=0
	echo $nfile
    for i in $(seq 1 $niter);
    do
        #echo echo $i >> $file_bash
        #echo $nmerge
        last=$nmerge
        nmerge=$(ceildiv $nmerge 2)
        $(merge_fst $nmerge $last $file_name $file_bash $task)
        for j in $(seq 1 $nmerge);
        do
            echo 'mv '$file_name'.'$j'.new '$file_name'.'$j' &' >> $file_bash
        done
        echo 'wait' >> $file_bash
    done
	echo 'mv '$file_name'.1 '$file_name >> $file_bash
}


