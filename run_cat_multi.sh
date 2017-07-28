cat_cmd='cat'
for i in 0 30000 40000 50000 65000 70000 75000 80000 85000 90000 100000 120000 135000 140000 145000 150000 165000;
do
    start=$i
    #middle=$(($i + 2500))
    #end=$(($i + 5000))
    end=$(($i + 5000))
    echo $start, $middle, $end
    cat_cmd=$cat_cmd' group.ec.txt.'$start'_'$end
    #cat_cmd=$cat_cmd' group.em3.txt.'$start'_'$middle' group.em3.txt.'$middle'_'$end
done
cat_cmd=$cat_cmd' > group.ec.txt'
echo $cat_cmd
