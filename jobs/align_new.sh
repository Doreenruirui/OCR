#! /bin/sh

trainFullPath=$1

trainFile=`basename $trainFullPath`
trainDir=`dirname $trainFullPath`

doAlign=True

# need to specify path to the M2M aligner
# EDIT THIS TO YOUR LOCAL PATH
#path2M2M=~/projects/OCR/OCR/m2m-aligner-master/
path2M2M=/home/rui/Package/m2m-aligner/

# we replace | to PIPE. Be sure to change that back
sed -e 's/|/PIPE/g' ${trainFullPath} > ${trainFullPath}.nopipe
sed -e 's/_/SPACE/g' ${trainFullPath}.nopipe > ${trainFullPath}.nodash
sed -e 's/ /_/g' ${trainFullPath}.nodash > ${trainFullPath}.nospace


if [ $doAlign = True ]; then
  $path2M2M/m2m-aligner --inFormat news --sepInChar "_MYJOIN_" --sepChar "|" -i ${trainFullPath}.nospace --maxX 1 --maxY 2 --delX --nullChar "EMPTY"
fi

alignFile="${trainFile}.nopipe.m-mAlign.1-2.delX.1-best.conYX.align"
alignFullPath=$trainDir/$alignFile

./removeLast.py < $alignFullPath > ${alignFullPath}.rl

# that's a hack
# Marmot doesn't like ":" and we also replace "-"
sed -i -e 's/:/COLON/g' ${alignFullPath}.rl
sed -i -e 's/-/DASH/g' ${alignFullPath}.rl # insert this in
