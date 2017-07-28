for i in 0 30000 40000 50000 65000 70000 75000 80000 85000 90000 100000 120000 135000 140000 145000 150000 165000;
do
    start=$i
    end=$(($i + 5000))
    echo $start, $end
    python evaluate_error_rate_2.py richmond/0/0/50/train_new group 100_35_group 100 $start $end 
done
#python evaluate_error_rate.py richmond/0/0/25/train test 100_20 100 260000 26
