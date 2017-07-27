for i in 140000 145000 150000 165000;
do
    start=$i
    end=$(($i + 5000))
    echo $start, $end
    python evaluate_error_rate_group.py richmond/0/0/50/train_new group 100_35_group 100 $start $end 
done
#python evaluate_error_rate.py richmond/0/0/25/train test 100_20 100 260000 26
