import argparse
import sys
import yt
import numpy as np

from scipy.spatial import KDTree

# Global parameter settings
#fields        = [("gas", "density"), ("gamer", "Phase")]  # Fields to plot
fields        = [("gas", "density"), ("gamer", "unwrapped_phase"), ("gas", "points_per_wavelength")]  # Fields to plot
axes          = ["x", "y", "z"]  # Directions
#axes          = ["x"]  # Directions
zoom_levels   = [1, 3, 5, 15, 20, 50, 100, 150]  # Zoom levels
#zoom_levels   = [1, 3, 5]  # Zoom levels
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
font_size             = 32

if plot_phase_mod_two_pi:

    def calculate_phase_mod_two_pi(field, data):
        return np.arctan2(np.sin(data["gamer", "Phase"]), np.cos(data["gamer", "Phase"]))

    yt.add_field(("gas", "Phase"), function=calculate_phase_mod_two_pi, sampling_type="local", units="")
    fields[1] = ("gas", "Phase")


def generate_kdtree_and_phases(ds):
    positions = []
    phases = []


    for grid in ds.index.grids:
        if grid.Level == 3:
            dds = grid.dds
            nx, ny, nz = grid.ActiveDimensions

            x = (np.arange(nx) ) * dds[0] + grid.LeftEdge[0]
            y = (np.arange(ny) ) * dds[1] + grid.LeftEdge[1]
            z = (np.arange(nz) ) * dds[2] + grid.LeftEdge[2]

            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

            positions.append(np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]))
            phases.append(grid["gamer", "Phase"].v.ravel())

    positions = np.vstack(positions)
    phases = np.hstack(phases)

    # Create a KDTree for the positions
    kdtree = KDTree(positions)

    # Debug output
    #print(f"KDTree constructed for dataset {ds}")
    #print(f"Parent positions shape: {positions.shape}")
    #print(f"Parent phases shape: {phases.shape}")
    #print(f"Sample parent positions: {positions[:10]}")
    #print(f"Sample parent phases: {phases[:10]}")

    return kdtree, phases
def unwrap_phase(field, data, kdtree, parent_phases):
    level = data["index", "grid_level"].astype(int)
    phase = data["gamer", "Phase"].copy()

    unwrapped_phase = phase.copy()

    # Use the KDTree and parent phases for levels >= 4
    max_level = int(level.max())
    if max_level >= 4:
        for l in range(4, max_level + 1):
            #print(f"Processing level {l}")
            mask = level == l
            #print(f"Number of cells at level {l}: {mask.sum()}")

            positions = np.column_stack((data["index", "x"][mask],
                                         data["index", "y"][mask],
                                         data["index", "z"][mask]))

            # Debug output for positions
            #print(f"Positions shape at level {l}: {positions.shape}")
            #print(f"Sample positions at level {l}: {positions[:10]}")

            # Find the nearest neighbors in the parent level
            distances, indices = kdtree.query(positions)
            closest_parent_phases = parent_phases[indices]

            # Debug output for nearest neighbors
            #print(f"Distances to parent cells at level {l}: {distances[:10]}")
            #print(f"Indices of parent cells at level {l}: {indices[:10]}")
            #print(f"Closest parent phases at level {l}: {closest_parent_phases[:10]}")

            # Unwrap phases
            delta_phase = unwrapped_phase[mask] - closest_parent_phases
            unwrapped_phase[mask] -= np.round(delta_phase / (2 * np.pi)) * 2 * np.pi

            # Debug output for delta phases
            #print(f"Delta phase before unwrapping at level {l}: {delta_phase[:10]}")
            #print(f"Delta phase after unwrapping at level {l}: {unwrapped_phase[mask][:10] - closest_parent_phases[:10]}")

    return unwrapped_phase

def add_unwrapped_phase_field(ds, kdtree, parent_phases):
    def get_unwrapped_phase(field, data):
        return unwrap_phase(field, data, kdtree, parent_phases)

    ds.add_field(("gas", "unwrapped_phase"), function=get_unwrapped_phase, sampling_type="local", units="")

def add_gradient_field(ds):
    ds.add_gradient_fields(("gas", "unwrapped_phase"))

    def compute_gradient(field, data):

        # Get the cell widths (dds) for this grid
        #dds = np.sqrt(data["index", "dx"]**2 + data["index", "dy"]**2 + data["index", "dz"**2])
        dds = data["index", "dx"]
        # Get unwrapped phase data
        magnitude = data[("gas", "unwrapped_phase_gradient_magnitude")]

        # Calculate gradients in each direction
        return 2*np.pi/(magnitude *dds + 0.0001)

    ds.add_field(("gas", "points_per_wavelength"), function=compute_gradient, sampling_type="cell", units="")



yt.enable_parallelism()

ts = yt.DatasetSeries([prefix + '/Data_%06d' % idx for idx in range(idx_start, idx_end + 1, didx)])
for ds in ts.piter():
    num = '%s' % ds
    num = int(num[5:11])

    ad = ds.all_data()
    ad.max_level = max_amr_level



    # Generate KDTree and parent phases for the current dataset
    kdtree, parent_phases = generate_kdtree_and_phases(ds)

    # Add the unwrapped phase field
    add_unwrapped_phase_field(ds, kdtree, parent_phases)
    add_gradient_field(ds)

    loc = ad.quantities.max_location('density')[1:]
    print("Location : ", loc)

    for zoom in zoom_levels:
        for ax in axes:
            for field in fields:
                if field[1] == "Phase" and plot_phase_mod_two_pi:
                    field = ("gas", "unwrapped_phase")
                sz = yt.SlicePlot(ds, ax, field, center=loc, data_source=ad)
                sz.hide_axes()
                sz.set_cmap(field, colormap)
                if field[1] == "Phase" or field[1] == "unwrapped_phase" or field[1] == "phase_gradient":
                    sz.set_log(field, False)
                sz.zoom(zoom)
                if field[1] == "density":
                    sz.set_zlim(field, 1.0e-31, 1.0e-22)

                # Set font sizes
                sz.set_font({'size': font_size})
                sz.annotate_scale(corner='lower_right', text_args={'size': font_size})

                sz.save('Data_%06d_Lv_%02d_Slice_%s_%s_x%d.png' % (num, max_amr_level, ax, field[1], zoom),
                        mpl_kwargs={"dpi": dpi})
                sz.annotate_grids(min_level=4, edgecolors="white")
                sz.save('Data_%06d_Lv_%02d_Slice_%s_%s_grid_x%d.png' % (num, max_amr_level, ax, field[1], zoom),
                        mpl_kwargs={"dpi": dpi})


