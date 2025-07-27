import os
import pandas as pd

def read_performance_file(file_path):
    """Read a Record__Performance file, ignoring lines starting with '#'."""
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
    if first_line.startswith('#'):
        columns = first_line[1:].split()
    else:
        return pd.DataFrame()
    
    # Read the file with the correct column names, skipping the first line
    data = pd.read_csv(file_path, delim_whitespace=True, comment='#', names=columns, skiprows=1)
    return data

base_dir = './'  # Update this path to your base directory
combined_data = pd.DataFrame()

for first_level_folder in os.listdir(base_dir):
    first_level_path = os.path.join(base_dir, first_level_folder)
    if os.path.isdir(first_level_path):  # Ensure it's a directory
        for rank_folder in os.listdir(first_level_path):
            rank_folder_path = os.path.join(first_level_path, rank_folder)
            if os.path.isdir(rank_folder_path) and rank_folder.startswith('rank_'):
                file_path = os.path.join(rank_folder_path, 'Record__Performance')
                if os.path.exists(file_path):
                    data = read_performance_file(file_path)
                    if not data.empty:
                        # Extract rank number from folder name and convert to integer
                        rank = int(rank_folder.split('_')[1])
                        data['Scheme'] = first_level_folder
                        data['Rank'] = rank
                        combined_data = pd.concat([combined_data, data], ignore_index=True)

# Save the combined data to a CSV file
output_file = 'strong_scaling_gramfe.csv'  # Update this path
combined_data.to_csv(output_file, index=False)

print("Data combined and saved to strong_scaling_gramfe.csv successfully.")

