source /home/dong.r/.bash_profile
python decode_line.py \
    --data_dir='/scratch/dong.r/Dataset/OCR/data/char_date_50_0/train/' \
    --train_dir='/scratch/dong.r/Model/OCR/nlc/char_date_50_0/train/'  \
    --start=$1 \
    --end=$2 \
    --beam_size=1000 \
    --gpu_frac=1 \
    --nthread=40 \
