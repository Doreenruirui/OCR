for i in $(seq 1 $1);
do 
    sbatch script/$2/run.sbatch.$i
done
