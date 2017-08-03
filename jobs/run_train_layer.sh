source /home/dong.r/.bash_profile
source activate tensorflow 
python train.py \
    --data_dir='/scratch/dong.r/Dataset/OCR/'$1 \
    --train_dir='/scratch/dong.r/Model/OCR/nlc/'$2 \
    --dev=$3 \
    --size=$4 \
    --num_layers=$5 \
    --print_every=200 
source deactivate
