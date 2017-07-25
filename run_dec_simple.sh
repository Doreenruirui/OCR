source /home/dong.r/.bash_profile
source activate tensorflow 
python decode_line_simple.py \
    --data_dir='/scratch/dong.r/Dataset/OCR/'$1 \
    --train_dir='/scratch/dong.r/Model/OCR/nlc/'$2 \
    --out_dir='/scratch/dong.r/Dataset/OCR/'$1'/'$3 \
    --beam_size=1000 \
    --dev=$4 \
    --start=$5 \
    --end=$6 \
    --print_every=200 
source deactivate
