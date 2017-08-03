folder_data=$1
file_no=$2
start=$3
size=$4
nline=$5
file_voc='/scratch/dong.r/Dataset/OCR/voc/ascii.syms'
end=$(($start  + $size))
end=$( (( $nline <= $end)) && echo "$nline" || echo "$end" )
start=$(($start + 1))
sed -n ''$start','$end'p' $folder_data'/train.text' > $folder_data'/train.text.'$file_no
#echo $folder_data'/train.text'
farcompilestrings -token_type=utf8 -keep_symbols=1 $folder_data'/train.text.'$file_no > $folder_data'/train.far.'$file_no
ngramcount -order=5  --require_symbols=false $folder_data'/train.far.'$file_no > $folder_data'/train.cnt.'$file_no
fstsymbols --isymbols=$file_voc --osymbols=$file_voc $folder_data'/train.cnt.'$file_no > $folder_data'/train_symbols.cnt.'$file_no
rm $folder_data'/train.text.'$file_no
rm $folder_data'/train.far.'$file_no
rm $folder_data'/train.cnt.'$file_no
