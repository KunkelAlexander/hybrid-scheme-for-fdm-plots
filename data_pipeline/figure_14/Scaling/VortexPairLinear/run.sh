#!/bin/bash

root_dir="./" # Change this to your root directory path
original_folder="template" # The folder to copy

for n in 16 20; do
    new_folder="rank_${n}" # Name of the new folder
    cp -r "${root_dir}/${original_folder}" "${root_dir}/${new_folder}" # Copy the original folder to the new one
    
    # Path to the submit_spock.job file in the new folder
    job_file="${root_dir}/${new_folder}/submit_spock.job"
    
    # Modify the submit_spock.job file
    sed -i "/#PBS -l nodes=/c\#PBS -l nodes=${n}:ppn=32" $job_file
    

    # Change directory to the new folder
    cd "${root_dir}/${new_folder}"
    
    # Submit the job
    qsub submit_spock.job
    
    # Optionally, return to the previous directory (root_dir or another desired directory)
    cd ..

done

