source /home/dong.r/.bash_profile
python decode_line_lm.py \
    --data_dir='/scratch/dong.r/Dataset/OCR/data/char_date_25_0/train_test_dev/train/' \
    --train_dir='/scratch/dong.r/Model/OCR/nlc/char_date_25_0/train_test_dev/train/' \
    --lmfile='/scratch/dong.r/Dataset/OCR/data/char_date_25_0/text.arpa' \
    --alpha=0.01 \
    --start=$1 \
    --end=$2 \
    --beam_size=1000 \
    --gpu_frac=1 \
    --nthread=50 \
