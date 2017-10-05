source /home/dong.r/.bash_profile
source activate tensorflow 
cd /home/dong.r/OCR
python train_downscale.py \
    --data_dir='/scratch/dong.r/Dataset/OCR/'$1 \
    --train_dir='/scratch/dong.r/Model/OCR/nlc/'$2 \
    --num_layers=$3 \
    --size=$4 \
    --print_every=200 
source deactivate
