delete_empty() {
    folder=$1
    if [ -d "$folder" ] && [ -z "$(find "$folder" -mindepth 1 -print -quit)" ]; then
        # Folder is empty, delete it
        rmdir "$folder"
        echo "Deleted empty folder: $folder"
    fi
}

timestamp=$(date +'%Y_%m_%dT%H-%M-%S')
folder32="timings/${timestamp}_FP32"
folder64="timings/${timestamp}_FP64"

make

# Check if an argument is passed
if [ -z "$1" ]; then
    echo "No argument passed. Generating plots for both FP32 and FP64"
    mkdir -p $folder32
    mkdir -p $folder64
    ./build/main 32 -p $folder32
    ./build/main 64 -p $folder64

    python3 scripts/plotting.py --input_folder $folder32
    python3 scripts/plotting.py --input_folder $folder64
    python3 scripts/plotting.py --input_folder $folder32 --compare
    python3 scripts/plotting.py --input_folder $folder64 --compare
    # python3 scripts/plotting.py --input_folder $folder32 --speedup
    # python3 scripts/plotting.py --input_folder $folder64 --speedup
    python3 scripts/plotting.py --input_folder $folder32 --precision
    python3 scripts/plotting.py --input_folder $folder64 --precision
else
    # Argument passed (either 32 or 64), generate plots for the specific one
    if [ "$1" -eq 32 ]; then
        echo "Generating FP32 plots"
        mkdir -p $folder32
        ./build/main 32 -p $folder32
        python3 scripts/plotting.py --input_folder $folder32
        python3 scripts/plotting.py --input_folder $folder32 --compare
        # python3 scripts/plotting.py --input_folder $folder32 --speedup
        python3 scripts/plotting.py --input_folder $folder32 --precision
    elif [ "$1" -eq 64 ]; then
        echo "Generating FP64 plots"
        mkdir -p $folder64
        ./build/main 64 -p $folder64
        python3 scripts/plotting.py --input_folder $folder64
        python3 scripts/plotting.py --input_folder $folder64 --compare
        # python3 scripts/plotting.py --input_folder $folder64 --speedup
        python3 scripts/plotting.py --input_folder $folder64 --precision
    else
        echo "Invalid argument. Please provide 32 or 64."
    fi
fi

delete_empty "$folder32"
delete_empty "$folder64"
