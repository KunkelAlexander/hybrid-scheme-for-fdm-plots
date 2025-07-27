#!/bin/bash

# Define the path to the folder containing the executables
executables_path="."

# Define Input__Parameter file path
input_file="Input__Parameter"

# List of executables
executables=("fd_2" "fd_4" "gramfe_fft" "gramfe_matmul")
# Function to modify Input__Parameter
modify_input() {
    # Set boundary conditions and ELBDM_BASE_SPECTRAL
    sed -i 's/OPT__BC_FLU_XM[[:space:]]\+[0-9]\+/OPT__BC_FLU_XM                '$1'/' "$input_file"
    sed -i 's/OPT__BC_FLU_XP[[:space:]]\+[0-9]\+/OPT__BC_FLU_XP                '$1'/' "$input_file"
    sed -i 's/OPT__BC_FLU_YM[[:space:]]\+[0-9]\+/OPT__BC_FLU_YM                '$1'/' "$input_file"
    sed -i 's/OPT__BC_FLU_YP[[:space:]]\+[0-9]\+/OPT__BC_FLU_YP                '$1'/' "$input_file"
    sed -i 's/OPT__BC_FLU_ZM[[:space:]]\+[0-9]\+/OPT__BC_FLU_ZM                '$1'/' "$input_file"
    sed -i 's/OPT__BC_FLU_ZP[[:space:]]\+[0-9]\+/OPT__BC_FLU_ZP                '$1'/' "$input_file"
    sed -i 's/ELBDM_BASE_SPECTRAL[[:space:]]\+[0-9]\+/ELBDM_BASE_SPECTRAL           '$2'/' "$input_file"
}

./clean.sh
rm -r results
mkdir results

modify_input 1 0

# Loop through each executable
for exe in "${executables[@]}"; do
    echo "Processing $exe..."

    # Call change_parameters.py with the executable
    python3 change_parameters.py -e "$executables_path/$exe"
done



# Set all boundary conditions to 1 and ELBDM_BASE_SPECTRAL to 0
modify_input 1 0

# Run change_parameters.py with base_fft
echo "Processing base_fft..."
modify_input 1 1
python3 change_parameters.py -e "$executables_path/base_fft"

# Set boundary conditions to 4 and ELBDM_BASE_SPECTRAL to 0 for fluid
modify_input 4 0
echo "Processing fluid..."
python3 change_parameters.py -e "$executables_path/fluid"

echo "All processes completed."

