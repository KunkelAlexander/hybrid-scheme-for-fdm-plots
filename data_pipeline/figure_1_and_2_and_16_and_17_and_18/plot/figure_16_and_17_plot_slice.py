import argparse
import sys
import yt
import numpy as np

# Global parameter settings
#fields        = [("gas", "density"), ("gamer", "Phase")]  # Fields to plot
fields        = [("gas", "density"), ("gamer", "Phase")]  # Fields to plot
axes          = ["x", "y"]  # Directions
zoom_levels   = [1, 5, 20, 100]#1, 3, 5, 15, 20, 50, 100, 150]  # Zoom levels
max_amr_level = 10  # Maximum level for sampling AMR grid
dpi           = 300
colormap      = 'magma'


# Load the command-line parameters
parser = argparse.ArgumentParser(description='Plot slices around halo')

# Define command-line arguments
parser.add_argument('-i', action='store', required=False, type=str, dest='prefix',
                    help='path prefix [%(default)s]', default='./')
parser.add_argument('-s', action='store', required=True, type=int, dest='idx_start',
                    help='first data index')
parser.add_argument('-e', action='store', required=True, type=int, dest='idx_end',
                    help='last data index')
parser.add_argument('-d', action='store', required=False, type=int, dest='didx',
                    help='delta data index [%(default)d]', default=1)
parser.add_argument('-m', action='store', required=False, type=int, dest='plot_phase_mod_two_pi',
                    help='plot phase field modulo 2 pi [%(default)d]', default=0)

args = parser.parse_args()

# Print the command-line arguments for reference
print('\nCommand-line arguments:')
print('-------------------------------------------------------------------')
for t in range(len(sys.argv)):
    print(str(sys.argv[t]))
print('')
print('-------------------------------------------------------------------\n')

idx_start             = args.idx_start
idx_end               = args.idx_end
didx                  = args.didx
prefix                = args.prefix
plot_phase_mod_two_pi = args.plot_phase_mod_two_pi
font_size   = 24
time_size   = 24

if plot_phase_mod_two_pi:

   def calculate_phase_mod_two_pi(field, data):
      return np.arctan2(np.sin(data["gamer", "Phase"]), np.cos(data["gamer", "Phase"]))

   yt.add_field(("gas", "Phase"), function=calculate_phase_mod_two_pi, sampling_type="local", units="")
   fields[1] = ("gas", "Phase")

yt.enable_parallelism()

ts = yt.DatasetSeries([prefix + '/Data_%06d' % idx for idx in range(idx_start, idx_end + 1, didx)])

for ds in ts.piter():
    num = '%s' % ds
    num = int(num[5:11])

    ad = ds.all_data()



    # Define the location and radius of the region of interest
    center = [0.67556152, 3.7842041, 2.85007324]  # Manually specified location
    radius = (0.4, 'unitary')  # Specify a radius (adjust as needed)
    
    # Create a spherical region around the specified location
    sphere = ds.sphere(center, radius)
    
    # Find the maximum density in the sphere
    max_density = sphere.max("density")
    max_location = sphere.quantities.max_location("density")
    
    # Print the results
    print(f"Maximum density in the region: {max_density}")
    print(f"Location of maximum density: {max_location}")

    loc = max_location[1:]
    print("Location : ", loc)

    for zoom in zoom_levels:
        if zoom == 1: 
            ad.max_level = 5
        else:
            ad.max_level = 8 
        for ax in axes:
            for field in fields:
                sz = yt.SlicePlot(ds, ax, field, center=loc, data_source=ad)
                sz.hide_axes()
                #sz.annotate_scale()
                sz.set_cmap(field, colormap)
                if field[1] == "Phase":
                    sz.set_log(field, False)
                sz.zoom(zoom)
                if field[1] == "density":
                    sz.set_unit(("gas", "density"), "Msun/kpc**3")
                    sz.set_zlim( field, 1.0e+0, 1.0e+7 )
                    # Modify the colorbar font size directly
                cbar = sz.plots[field].cb
                cbar.ax.tick_params(labelsize=font_size)


                # Set colorbar font size
                cbar = sz.plots[field].cb
                cbar.ax.tick_params(labelsize=time_size)
                
                # Set general font size
                sz.set_font({'size': time_size})
                
                if zoom != 10 and zoom != 15:
                    # Annotate scale with custom size
                    sz.annotate_scale(pos=(0.85, 0.05), corner='lower_right', text_args={'size': time_size})

                    # Annotate timestamp with custom size
                    sz.annotate_timestamp(time_unit='Gyr', redshift=True, corner='upper_right', text_args={'color': 'white', 'size': time_size})


                sz.save('Data_%06d_Lv_%02d_Slice_%s_%s_x%d.png' % (num, max_amr_level, ax, field[1], zoom),
                        mpl_kwargs={"dpi": dpi})
                if zoom == 1:
                    sz.annotate_grids(min_level=4)
                else:
                    sz.annotate_grids(min_level=4)
                sz.save('Data_%06d_Lv_%02d_Slice_%s_%s_wave_grid_x%d.png' % (num, max_amr_level, ax, field[1], zoom),
                        mpl_kwargs={"dpi": dpi})
                if zoom == 1:
                    sz.annotate_grids(min_level=0)
                else:
                    sz.annotate_grids(min_level=0)
                sz.save('Data_%06d_Lv_%02d_Slice_%s_%s_grid_x%d.png' % (num, max_amr_level, ax, field[1], zoom),
                        mpl_kwargs={"dpi": dpi})
