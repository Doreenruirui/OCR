source /home/dong.r/.bash_profile
source activate tensorflow 
cd /home/dong.r/OCR
python train.py \
    --data_dir='/scratch/dong.r/Dataset/OCR/'$1 \
    --train_dir='/scratch/dong.r/Dataset/OCR/'$1'/model' \
    --dev=$2 \
    --size=$3 \
    --print_every=200 
source deactivate
