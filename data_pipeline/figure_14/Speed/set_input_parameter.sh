#!/bin/bash

# Loop over all Input__Parameter files in subdirectories of the current directory
find . -type f -name "Input__Parameter" | while read file; do
    # Set OMP_NTHREAD to 8
    sed -i 's/^\(OMP_NTHREAD[[:space:]]*\).*$/\1 8          # number of OpenMP threads (<=0=auto -> omp_get_max_threads) [-1] ##OPENMP ONLY##/' "$file"
    
    # Set END_STEP to 20
    sed -i 's/^\(END_STEP[[:space:]]*\).*$/\1 20          # end step (<0=auto -> must be set by test problems or restart) [-1]/' "$file"

    # Set NX0_TOT_X, NX0_TOT_Y, NX0_TOT_Z to 1024
    sed -i 's/^\(NX0_TOT_X[[:space:]]*\).*$/\1 1024         # number of base-level cells along x/' "$file"
    sed -i 's/^\(NX0_TOT_Y[[:space:]]*\).*$/\1 1024         # number of base-level cells along y/' "$file"
    sed -i 's/^\(NX0_TOT_Z[[:space:]]*\).*$/\1 1024         # number of base-level cells along z/' "$file"

    # Set OUTPUT_STEP to 100
    sed -i 's/^\(OUTPUT_STEP[[:space:]]*\).*$/\1 100          # output data every OUTPUT_STEP step ##OPT__OUTPUT_MODE==1 ONLY##/' "$file"

    sed -i 's/^\(REGRID_COUNT[[:space:]]*\).*$/\1 100          # output data every OUTPUT_STEP step ##OPT__OUTPUT_MODE==1 ONLY##/' "$file"

    echo "Modified: $file"
done

