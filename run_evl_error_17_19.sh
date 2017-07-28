for i in $(seq 17 19);
do
    j=$(($i - 1))
    start=$(($j * 10000))
    end=$(($i * 10000))
    python evaluate_error_rate.py richmond/0/0/25/train test 100_20 100 $start $end 
done
#python evaluate_error_rate.py richmond/0/0/25/train test 100_20 100 260000 26
