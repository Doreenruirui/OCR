source /home/dong.r/.bash_profile
source activate tensorflow
python decode_line_lm_combine.py \
    --data_dir='/scratch/dong.r/Dataset/OCR/data/char_date_25_0_new/train/' \
    --train_dir='/scratch/dong.r/Model/OCR/nlc/char_date_25_0_new/train/' \
    --start=$1 \
    --end=$2 \
    --beam_size=100 \
    --gpu_frac=1 \
    --nthread=40 
source deactivate
