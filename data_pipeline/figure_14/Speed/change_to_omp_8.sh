#!/bin/bash

# Loop over all submit_spock.job files in subdirectories of the current directory
find . -type f -name "submit_spock.job" | while read file; do
    # Use grep with a pattern that matches any executable name
    if grep -q "mpirun -map-by ppr:1:socket:pe=32 --report-bindings .*/.* 1>>log 2>&1" "$file"; then
        # Use sed to replace the parameter in-place for any executable name
        sed -i 's/mpirun -map-by ppr:1:socket:pe=32 --report-bindings/mpirun -map-by ppr:4:socket:pe=8 --report-bindings/g' "$file"
        echo "Modified: $file"
    fi
done

