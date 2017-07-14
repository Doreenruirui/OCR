source /home/dong.r/nlc-master/bash_function.sh

write_fst(){
    local line=$1
    #echo $line
    line=${line//$'\n'/}
    local file_fst=$2
    local lam=$3
	local string=$(echo -e "$line" | cut -d $'\t' -f1)
    local weight=$(echo -e "$line" | cut -d $'\t' -f2)
	local weight=$(bc -l <<< "$weight*$lam")
    local weight=$(bc -l <<< "$weight*-1")
    local cur_char=''
	local j=0
	local cur_w=0
	local cur_arc=''
	$(rm_file $file_fst)
    local len=${#string}
    for i in $(seq 1 $len);
    do
        j=$(($i - 1))
        cur_char=${string:$j:1}
        #echo "$cur_char"
        cur_char=$( [[ "$cur_char" == [[:space:]] ]] && echo '<space>' || echo "$cur_char" )
        #echo "$cur_char"
        cur_w=$( (( $i == 1)) && echo $weight || echo '0' )
        cur_arc=$j$'\t'$i$'\t'"$cur_char"$'\t'"$cur_char"$'\t'$cur_w
        #echo "$cur_arc"
        echo "$cur_arc" >> $file_fst
    done
    echo -e $len'\t'0 >> $file_fst
    }


write_fst_batch(){
    local start=$1
    local end=$2
    local file_name=$3
    local file_fst=$4
    local lam=$5
	local file_bash=$6
	local file_sym=$7
    local cur_str=''
	local cur_fst_file=''
	local cur_id=0
	local nline=$(($end + 1 - $start))
    echo $nline
    for i in $(seq 1 $nline);
    do
        cur_id=$(($start + $i - 1))
        cur_str='sed -n '$cur_id','$cur_id'p '$file_name
		cur_fst_file=$file_fst'.'$i
		echo 'write_fst "$('$cur_str')" '$cur_fst_file' '$lam' &' >> $file_bash
    done
    echo wait >> $file_bash
    for i in $(seq 1 $nline);
    do
		cur_fst_file=$file_fst'.'$i
		echo 'fstcompile --isymbols='$file_sym' --osymbols='$file_sym' --keep_isymbols --keep_osymbols '$cur_fst_file' '$cur_fst_file' & ' >> $file_bash
    done
    echo wait >> $file_bash
    }

get_fst_for_line(){
    local line_no=$1
    local file_fst=$2
    local file_name=$3
    local file_bash=$4
    local lam=$5
	local file_sym=$6
    local start=$(($(($(($line_no - 1)) * 100)) + 1))
    local end=$(($line_no * 100))
    #echo $start, $end >> $file_bash
    #local a=$(sed -n ''$start','$end'p' $file_name)
	#echo ${#a}
    #local task=('fstunion ' ' ' ' | fstrmepsilon - | fstdeterminize - | fstminimize -'):
    #echo 'source /home/dong.r/nlc-master/bash_fst.sh'
    $(rm_file $file_bash)
    echo 'source /home/dong.r/nlc-master/bash_fst.sh' >> $file_bash
    $(write_fst_batch $start $end $file_name $file_fst $lam $file_bash $file_sym)
    #echo write_fst_batch $start $end $file_name $file_fst $lam $file_bash $file_sym
    #echo merge_iter 100 "$task" $file_bash $file_fst
    $(merge_iter 100 1 $file_bash $file_fst)
}

get_fst_for_group(){
	local start=$1
    local end=$2
    local folder_data='/scratch/dong.r/Dataset/OCR/'$3
	local file_name=$4
	#echo $file_name
    local lam=$5
	local file_inter='/scratch/dong.r/Dataset/lm/concat.fst'
    local file_no=$(echo -e "$file_name" | cut -d $'_' -f1 | cut -d $'.' -f4)
    echo $file_no
    file_name=$folder_data'/'$file_name
    local file_bash='/home/dong.r/nlc-master/run_group.'$file_no'.'$start
    echo $file_bash
    local file_lm='/scratch/dong.r/Dataset/OCR/lm/char/richmond/train.mod'
	local file_sym='/scratch/dong.r/Dataset/OCR/voc/ascii.syms'
	local file_fst=$folder_data'/fst.tmp'
	$(rm $file_bash)
    echo 'source /home/dong.r/nlc-master/bash_fst.sh' >> $file_bash
	for i in $(seq $start $end);
    do
		$(rm $file_bash'.'$i)
        cur_id=$(($i -$start + 1))
        cur_file_fst=$file_fst'.'$file_no'.'$start'.'$cur_id
        echo 'get_fst_for_line '$i' '$cur_file_fst' '$file_name' '$file_bash'.'$i' '$lam' '$file_sym'  &'>> $file_bash
    done
	echo wait >> $file_bash
    for i in $(seq $start $end);
    do
        cur_id=$(($i -$start + 1))
        cur_file_fst=$file_fst'.'$file_no'.'$start'.'$cur_id
        echo 'sh '$file_bash'.'$i' &' >> $file_bash
    done
    echo wait >> $file_bash
    for i in $(seq $start $end);
    do
        echo 'rm '$file_bash'.'$i' &' >> $file_bash
    done
    echo wait >> $file_bash
    nline=$(($end + 1 - $start))
	echo $nline
    $(merge_iter $nline 2 $file_bash $file_fst'.'$file_no'.'$start)
	echo 'fstintersect '$file_fst'.'$file_no'.'$start' '$file_lm' | fstshortestpath - | fstreverse - | fstrmepsilon - > '$file_fst'.'$file_no'.'$start'.score' >> $file_bash	
}

