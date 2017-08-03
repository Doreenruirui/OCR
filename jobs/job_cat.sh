nfile=$1
size=$2
file_prefix=$3
cat_cmd='cat'
for i in $(seq 1 $nfile);
do    
    j=$(($i - 1))
    start=$(($j * $size))
    end=$(($i * $size))
    cat_cmd=$cat_cmd' '$file_prefix'.'$start'_'$end
done
cat_cmd=$cat_cmd' > '$file_prefix
echo $cat_cmd
