input=$1
input=$(echo "$input" | awk '{print tolower($0)}')
input="${input//\ /<space>}"
echo $input
