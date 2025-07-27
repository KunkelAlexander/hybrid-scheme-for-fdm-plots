#!/bin/bash

# Loop over all Input__Parameter files in subdirectories of the current directory
find . -type f -name "Input__Parameter" | while read file; do
    # Set OMP_NTHREAD to 8
    sed -i 's/^\(OMP_NTHREAD[[:space:]]*\).*$/\1 8          # number of OpenMP threads (<=0=auto -> omp_get_max_threads) [-1] ##OPENMP ONLY##/' "$file"
    
    # Set END_STEP to 20
    sed -i 's/^\(END_STEP[[:space:]]*\).*$/\1 20          # end step (<0=auto -> must be set by test problems or restart) [-1]/' "$file"

    echo "Modified: $file"
done

