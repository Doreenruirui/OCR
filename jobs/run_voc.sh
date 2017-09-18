source /home/dong.r/.bash_profile
cd /home/dong.r/OCR/
python prepare_data.py \
        --data_dir='/scratch/dong.r/Dataset/OCR/'$1 \
        --gen_voc=1 \
