set -e

# Call ./fd_2 and move its results
python3 change_parameters.py  -e ./fd_2
python3 change_parameters.py  -e ./fd_4
python3 change_parameters.py  -e ./gramfe_fft

# Finally, call the Python script to extract results
# python3 extract_results.py

echo "All operations completed successfully."

