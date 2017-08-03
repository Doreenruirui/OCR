source /home/dong.r/OCR/bash_function.sh

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

