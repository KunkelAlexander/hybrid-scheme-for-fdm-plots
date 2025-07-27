import os
import pandas as pd

def read_performance_file(file_path):
    # Open the file and read the first line to capture column names
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
    # Check if the first line starts with '#', and remove '#' if it does
    if first_line.startswith('#'):
        columns = first_line[1:].split()
    else:
        # If there's no '#', assume the first line is not column names
        return pd.DataFrame()
    
    # Now read the file into a DataFrame using the extracted column names
    data = pd.read_csv(file_path, delim_whitespace=True, comment='#', names=columns, skiprows=1)
    return data

# Define the root directory containing the subfolders
root_dir = './'

# Prepare an empty DataFrame for collecting the data
combined_data = pd.DataFrame()

# Iterate over each subfolder in the root directory
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    
    # Check if the current item is a directory
    if os.path.isdir(subfolder_path):
        # Construct the path to the Record__Performance file within the subfolder
        file_path = os.path.join(subfolder_path, 'Record__Performance')
        
        # Check if the Record__Performance file exists
        if os.path.exists(file_path):
            # Read the data from the file
            data = read_performance_file(file_path)
            
            # Add a new column with the subfolder name
            data['FolderName'] = subfolder
            
            # Append the data from this subfolder to the combined DataFrame
            combined_data = pd.concat([combined_data, data], ignore_index=True)

# Write the combined data to a CSV file
combined_data.to_csv('performance_test.csv', index=False)

print("Data combined and saved to performance_test.csv successfully.")

