import os
import shutil
import matplotlib.pyplot as plt

# Define the zoom levels
zooms = [1, 5, 20, 100]

# Define the source directory and the target directory
source_dir = 'results/halo'
target_dir = 'results/to_plot'

# Create the target directory if it does not exist
os.makedirs(target_dir, exist_ok=True)

# Define the file patterns
file_patterns = [
    'Data_000045_Lv_10_Slice_z_density_x{}.png',
    'Data_000045_Lv_10_Slice_z_density_wave_grid_x{}.png',
    'Data_000045_Lv_10_Slice_z_Phase_x{}.png',
    'Data_000045_Lv_10_Slice_z_Phase_wave_grid_x{}.png'
]

# Loop through the zoom levels and file patterns, copying each file to the target directory
for zoom in zooms:
    for pattern in file_patterns:
        src_file = os.path.join(source_dir, pattern.format(zoom))
        dst_file = os.path.join(target_dir, pattern.format(zoom))
        shutil.copy(src_file, dst_file)
        print(f"Copied {src_file} to {dst_file}")

print("All files have been copied to the 'to_plot' directory.")
