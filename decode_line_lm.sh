source /home/dong.r/.bash_profile
source activate tensorflow
python -m pdb decode_line_lm_combine.py \
    --data_dir='/scratch/dong.r/Dataset/OCR/data/char_date_50_0_new/'$3 \
    --train_dir='/scratch/dong.r/Model/OCR/nlc/char_date_50_0_new/'$4 \
    --start=$1 \
    --end=$2 \
    --beam_size=100 \
    --gpu_frac=1 \
    --nthread=40 
source deactivate
