set -e

# Call ./fd_2 and move its results
./fd_2
sh move2folder.sh fd_2_results

# Call ./fd_4 and move its results
./fd_4
sh move2folder.sh fd_4_results

# Call ./gramfe_fft and move its results
./gramfe_fft
sh move2folder.sh gramfe_fft_results

# Finally, call the Python script to extract results
python3 extract_results.py

echo "All operations completed successfully."

