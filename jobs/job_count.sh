for i in $(seq 1 $1);
do
    #echo $i
    j=$(($i - 1))
    start=$(($j * $2))
    end=$(($i * $2))
    echo $(cat $3.$start'_'$end | wc -l)
    #echo $(cat man_wit.test.richmond.ec4.txt.$start'_'$end | wc -l)
    #echo $start, $end, $(cat group.om1.txt.$start'_'$end | wc -l) $(cat group.om2.txt.$start'_'$end | wc -l) $(cat group.ec3.txt.$start'_'$end | wc -l)
done
