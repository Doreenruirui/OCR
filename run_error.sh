source /home/dong.r/.bash_profile
source activate tensorflow 
python error.py \
    --data_dir='/scratch/dong.r/Dataset/OCR/'$1 \
    --train_dir='/scratch/dong.r/Model/OCR/nlc/'$2 \
    --print_every=200 
source deactivate
