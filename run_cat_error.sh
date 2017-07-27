cat_cmd='cat'
for i in $(seq 1 26);
do
    j=$(($i - 1))
    start=$(($j * 10000))
    end=$(($i * 10000))
    cat_cmd=$cat_cmd' test.ec.txt.'$start'_'$end 
done
cat_cmd=$cat_cmd' test.ec.txt.260000_268763 > test.ec.txt.0_268763'
echo $cat_cmd
#python evaluate_error_rate.py richmond/0/0/25/train test 100_20 100 260000 26
