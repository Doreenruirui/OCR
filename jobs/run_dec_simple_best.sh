source /home/dong.r/.bash_profile
source activate tensorflow 
cd /home/dong.r/OCR
python decode_line_simple_best.py \
    --data_dir='/scratch/dong.r/Dataset/OCR/'$1 \
    --train_dir='/scratch/dong.r/Model/OCR/nlc/'$2 \
    --out_dir='/scratch/dong.r/Dataset/OCR/'$1'/'$3 \
    --beam_size=100 \
    --dev=$4 \
    --start=$5 \
    --end=$6 \
    --num_layers=$7 \
    --size=$8 \
    --print_every=200
source deactivate
