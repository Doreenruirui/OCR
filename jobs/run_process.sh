python process_data.py $1 load
python process_data.py $1 error
python process_data.py $1 split test 0
python process_data.py $1 split dev 0 0
python process_data.py $1 write test 0
python process_data.py $1 write dev 0 0 50
