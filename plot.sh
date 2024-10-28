timestamp=$(date +'%Y_%m_%dT%H-%M-%S')
folder="timings/$timestamp"

mkdir -p $folder
make

./build/main -p $folder
#source ../.venv/bin/activate
#python3 utils/plotting.py --input_folder $folder
#python3 utils/plotting.py --input_folder $folder --compare
#python3 utils/plotting.py --input_folder $folder --speedup