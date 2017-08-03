source /home/dong.r/.bash_profile
source activate tensorflow 
python error_pre_copy.py \
    --data_dir='/scratch/dong.r/Dataset/OCR/data/char_date_50_0_new/'$1 \
    --train_dir='/scratch/dong.r/Model/OCR/nlc/char_date_50_0_new/'$2 \
    --print_every=200 
source deactivate
