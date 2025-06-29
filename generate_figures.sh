#!/bin/bash

# Define the main log file where all output will be appended
main_log="all_figures.log"

# Clear the log file before starting (optional)
> "$main_log"

for i in $(seq 1 18); do
    if [ "$i" -eq 2 ] || [ "$i" -eq 5 ]; then
        continue
    fi
    script="generate_figure_${i}.py"
    echo "Running $script" | tee -a "$main_log"

    # Run the script and append both stdout and stderr to the same log file
    python3 "$script" >> "$main_log" 2>&1

    # Add a separator to make the log easier to read
    echo "----------------------------------------" >> "$main_log"
done
