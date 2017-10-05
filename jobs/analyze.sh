#!/usr/bin/env bash
source /home/dong.r/OCR/jobs/bash_function.sh
folder=$1
file1=$2
file2=$3
file3=$5
lineno=$6
test_folder=$4
fileno=$(ceildiv $lineno 5000)
#echo $fileno
folder_data='/scratch/dong.r/Dataset/OCR/'
cur_folder=$folder_data$folder
echo X: 
echo $(sed -n "$lineno, $lineno p" $cur_folder$file1) 
echo Y:
echo $(sed -n "$lineno, $lineno p" $cur_folder$file2) 
start_no=$(($fileno - 1 ))
start=$(($start_no * 5000))
end=$(($fileno * 5000))
file3=$file3'.'$start'_'$end
cur_line=$(($lineno - $start - 1))
test_line=$((cur_line * 100 + 1))
end_line=$((test_line + 100))
#echo sed -n "$test_line, $end_line p" $file3
#echo $cur_line
#echo $test_line
echo OUR: 
#echo $(sed -n "$test_line, $test_line p" $cur_folder$test_folder$file3)
for i in $(seq $test_line $end_line);
do
    echo $(sed -n "$i, $i p" $cur_folder$test_folder$file3)
done
#echo sed -n '"'$test_line', '$end_line' p"' $file3
