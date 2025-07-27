#!/bin/bash

# List of executable names
executables=("o5")

# Loop through each executable
for executable_name in "${executables[@]}"; do
    # Run change_parameters.py
    python3 change_parameters.py -e "./${executable_name}"
    
    # Generate folder name
    folder_name="${executable_name}_OPT__FLU_INT_SCHEME_OPT__INT_PHASE_VorPairLin_kx_OPT__REF_FLU_INT_SCHEME"
    
    # Run summarise_error.py
    python3 summarise_error.py "${folder_name}" "${executable_name}.csv"
done

