import yt
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # For log scale in imshow


# Ensure Matplotlib uses the Agg backend, which is great for scripts
plt.switch_backend('agg')

def interpolation_error(exact, interpolated):
    return np.mean(np.abs(exact - interpolated))

def compute_l1_error_and_plot(ds_low_res, ds_high_res, fn):
    # Assuming both datasets have the same domain properties
    level = 1
    dims = ds_low_res.domain_dimensions
    left_edge = ds_low_res.domain_left_edge

    cg_low_res = ds_low_res.covering_grid(level=level, left_edge=left_edge, dims=dims*2)
    cg_high_res = ds_high_res.covering_grid(level=level, left_edge=left_edge, dims=dims*2)

    real_low_res = cg_low_res["Real"].to_ndarray()
    imag_low_res = cg_low_res["Imag"].to_ndarray()
    real_high_res = cg_high_res["Real"].to_ndarray()
    imag_high_res = cg_high_res["Imag"].to_ndarray()
    wave_function_interpolated =  real_low_res[:, :, 0] + 1j * imag_low_res[:, :, 0]
    hr_wave_function =  real_high_res[:, :, 0] + 1j * imag_high_res[:, :, 0]

    error = interpolation_error(hr_wave_function, wave_function_interpolated)
    # Visualize the re + im error
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log10(np.abs(hr_wave_function - wave_function_interpolated)), vmin = -6, vmax = 1)
    plt.colorbar(label=f'Density |ψ|^2 = {error:.2e}')
    plt.title('Interpolated Density Plot of 2D Schrödinger Equation Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(fn+"/error.png")
    plt.close()


    plt.figure(figsize=(10, 8))
    plt.plot(np.abs(hr_wave_function)[:, 0], lw = 3)
    plt.plot(np.abs(wave_function_interpolated)[:, 0])
    plt.yscale("log")
    plt.title('Interpolated Density Plot of 2D Schrödinger Equation Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(fn+"/dens.png")
    plt.close()
    
    
    
    plt.figure(figsize=(10, 8))
    plt.plot(np.angle(hr_wave_function)[:, 0], lw = 3)
    plt.plot(np.angle(wave_function_interpolated)[:, 0])
    plt.title('Interpolated Phase Plot of 2D Schrödinger Equation Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(fn+"/phase.png")
    plt.close()
    return error

def compute_l1_error(ds_low_res, ds_high_res):
    # Assuming both datasets have the same domain properties
    level = 0
    dims = ds_low_res.domain_dimensions
    left_edge = ds_low_res.domain_left_edge
    
    cg_low_res = ds_low_res.covering_grid(level=level, left_edge=left_edge, dims=dims)
    cg_high_res = ds_high_res.covering_grid(level=level, left_edge=left_edge, dims=dims)
    
    real_low_res = cg_low_res["Real"].to_ndarray()
    imag_low_res = cg_low_res["Imag"].to_ndarray()
    real_high_res = cg_high_res["Real"].to_ndarray()
    imag_high_res = cg_high_res["Imag"].to_ndarray()
    
    l1_error_real = np.mean(np.abs(real_low_res - real_high_res))
    l1_error_imag = np.mean(np.abs(imag_low_res - imag_high_res))
    
    return 0.5 * (l1_error_real + l1_error_imag)

if len(sys.argv) != 4:
    print("Usage: python compute_errors.py <base_dir> <folder_name> <output_file>")
    sys.exit(1)

base_dir = sys.argv[1]
folder_name = sys.argv[2]
output_file = sys.argv[3]

folder_path = os.path.join(base_dir, folder_name)
ds_low_res = yt.load(os.path.join(folder_path, 'Data_000001'))
ds_high_res = yt.load(os.path.join(folder_path, 'Data_000000_hr'))
fn = folder_path

mean_l1_error = compute_l1_error(ds_low_res, ds_high_res)

# Write the error to a specified output file
with open(output_file, 'w') as file:
    file.write(f"{mean_l1_error}\n")

