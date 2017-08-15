source ~/.bash_profile
cd ../
python prepare_data_multi.py \
        --data_dir='/scratch/dong.r/Dataset/OCR/'$1 \
        --gen_voc=1 \
