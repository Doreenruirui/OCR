source /home/dong.r/.bash_profile
python decode_line.py \
    --data_dir='/scratch/dong.r/Dataset/OCR/data/char_25/train_test/' \
    --train_dir='/scratch/dong.r/Model/OCR/nlc/char_25/train_test/' \
    --start=$1 \
    --end=$2 \
    --beam_size=100 \
    --gpu_frac=1 \
    --nthread=40 \
