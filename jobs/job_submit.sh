for i in $(seq $1 $2);
do 
    sbatch /home/dong.r/OCR/script/$3/run.sbatch.$i
done
