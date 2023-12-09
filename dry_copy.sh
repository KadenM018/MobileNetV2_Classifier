#!/bin/bash

# Define source and destination directories
source_dir="/home/greg/MobileNetV2_Classifier/data/ASL"
destination_dir="/home/greg/MobileNetV2_Classifier/run400/ASL"
n=400  # Set the value of n to the desired number of files to copy
ifs=$IFS
IFS=$'\
'
# Loop through each folder in the source directory
for folder in {0..9} {A..Z}; do
    # Construct the paths for source and destination folders
    source_folder="${source_dir}/${folder}"
    destination_folder="${destination_dir}/${folder}"

    # Check if the destination folder exists, if not, create it
    if [ ! -d "$destination_folder" ]; then
        mkdir -p "$destination_folder"
    fi

    # Get a list of files in the source folder
    files_in_folder=("$source_folder"/*)

    # Determine the number of files to copy
    num_files=${#files_in_folder[@]}
    files_to_copy=("${files_in_folder[@]}")

    # If there are not enough files, copy all available files
    if [ $num_files -lt $n ]; then
        echo "Not enough files in $source_folder to copy $n files. Copying all available files."
    else
        # Shuffle the list of files and get the first n
        shuffled_files=($(shuf -e "${files_in_folder[@]}"))
        files_to_copy=("${shuffled_files[@]:0:$n}")
    fi

    # Copy the selected files to the destination folder
    for file_to_copy in "${files_to_copy[@]}"; do
        cp -v "$file_to_copy" "$destination_folder"
    done
done
IFS=ifs


