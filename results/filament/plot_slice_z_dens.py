import argparse
import sys
import yt
import pandas as pd
import numpy as np
from scipy.spatial import KDTree

import matplotlib.colors as mcolors

# load the command-line parameters
parser = argparse.ArgumentParser( description='Slice of mass density' )

parser.add_argument( '-p', action='store', required=False, type=str, dest='prefix',
                     help='prefix [%(default)s]', default='../../' )
parser.add_argument( '-s', action='store', required=True,  type=int, dest='idx_start',
                     help='first data index' )
parser.add_argument( '-e', action='store', required=True,  type=int, dest='idx_end',
                     help='last data index' )
parser.add_argument( '-d', action='store', required=False, type=int, dest='didx',
                     help='delta data index [%(default)d]', default=1 )
parser.add_argument( '-halo', action='store', required=False, type=int, dest='halo',
                     help='which halo [%(default)d]', default=1 )


args=parser.parse_args()

idx_start   = args.idx_start
idx_end     = args.idx_end
didx        = args.didx
prefix      = args.prefix
halo        = args.halo

field       = 'density'
colormap    = 'magma'
center_mode = 'c'
dpi         = 300

font_size   = 32
time_size   = 40

def _real(field, data):
    return np.cos(data["gamer", "Phase"])

def _imag(field, data):
    return np.sin(data["gamer", "Phase"])

def _pm2(field, data):
    return np.arctan2(np.sin(data["gamer", "Phase"]), np.cos(data["gamer", "Phase"]))


import logging

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_kdtree_and_phases(ds):
    positions = []
    phases = []

    for grid in ds.index.grids:
        if grid.Level == 3:
            dds = grid.dds
            nx, ny, nz = grid.ActiveDimensions

            x = (np.arange(nx)) * dds[0] + grid.LeftEdge[0]
            y = (np.arange(ny)) * dds[1] + grid.LeftEdge[1]
            z = (np.arange(nz)) * dds[2] + grid.LeftEdge[2]

            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

            positions.append(np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]))
            phases.append(grid["gamer", "Phase"].v.ravel())

    positions = np.vstack(positions)
    phases = np.hstack(phases)

    # Create a KDTree for the positions
    kdtree = KDTree(positions)

    logger.debug(f"KDTree constructed for dataset {ds}")
    logger.debug(f"Parent positions shape: {positions.shape}")
    logger.debug(f"Parent phases shape: {phases.shape}")
    logger.debug(f"Sample parent positions: {positions[:10]}")
    logger.debug(f"Sample parent phases: {phases[:10]}")

    return kdtree, phases

def unwrap_phase(field, data, kdtree, parent_phases):
    level = data["index", "grid_level"].astype(int)
    try:
        phase = data["gamer", "Phase"].copy()
    except KeyError:
        raise KeyError("The field 'gamer', 'Phase' does not exist in the dataset.")

    unwrapped_phase = phase.copy()

    # Use the KDTree and parent phases for levels >= 4
    max_level = int(level.max())
    if max_level >= 4:
        for l in range(4, max_level + 1):
            logger.debug(f"Processing level {l}")
            mask = level == l
            logger.debug(f"Number of cells at level {l}: {mask.sum()}")

            positions = np.column_stack((data["index", "x"][mask],
                                         data["index", "y"][mask],
                                         data["index", "z"][mask]))

            logger.debug(f"Positions shape at level {l}: {positions.shape}")
            logger.debug(f"Sample positions at level {l}: {positions[:10]}")


            # Find the 8 nearest neighbors in the parent level
            distances, indices = kdtree.query(positions, k=8)

            for i, (dist, ind) in enumerate(zip(distances, indices)):
                if not np.allclose(dist, dist[0]):
                    logger.warning(f"Cell at position {positions[i]} does not have 8 equidistant parent cells.")
                    logger.warning(f"Parent positions: {kdtree.data[ind]}")
                    logger.warning(f"Distances to parent cells: {dist}")

            # Use the closest parent phase for each cell
            closest_parent_phases = parent_phases[indices[:, 0]]

            logger.debug(f"Sample distances to parent cells at level {l}: {distances[:10]}")
            logger.debug(f"Sample indices of parent cells at level {l}: {indices[:10]}")
            logger.debug(f"Sample closest parent phases at level {l}: {closest_parent_phases[:10]}")

            # Unwrap phases
            delta_phase = unwrapped_phase[mask] - closest_parent_phases
            unwrapped_phase[mask] -= np.round(delta_phase / (2 * np.pi)) * 2 * np.pi

            logger.debug(f"Delta phase before unwrapping at level {l}: {delta_phase[:10]}")
            logger.debug(f"Delta phase after unwrapping at level {l}: {unwrapped_phase[mask][:10] - closest_parent_phases[:10]}")

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





yt.add_field(
  name=("gamer", "Imag"),
  function=_imag,
  sampling_type="local",
  units="",
)

yt.add_field(
  name=("gamer", "Real"),
  function=_real,
  sampling_type="local",
  units="",
)


yt.add_field(
  name=("gas", "Phase"),
  function=_pm2,
  sampling_type="local",
  units="",
)

yt.enable_parallelism()


ts = yt.DatasetSeries( [ prefix+'/Data_%06d'%idx for idx in range(idx_start, idx_end+1, didx) ] )


for ds in ts.piter():

    num = '%s'%ds
    num = int(num[9:11])

    ad = ds.all_data()

    # Generate KDTree and parent phases for the current dataset
    kdtree, parent_phases = generate_kdtree_and_phases(ds)

    # Add the unwrapped phase field
    add_unwrapped_phase_field(ds, kdtree, parent_phases)
    add_gradient_field(ds)


    for j, coordinates in enumerate(np.linspace(0.0, 4.5, 20)[3:8]):
        for i, ax in enumerate(["x"]):

            field = ("gas", "density")
            center = [coordinates, ds.domain_width[1] * 3/4 * 1.2, ds.domain_width[2] * 3/4 * 1.065]
            sz_dens = yt.SlicePlot( ds, ax, field, center = center) #, center=[coordinate_x, coordinate_y, coordinate_z] )
            sz_dens.set_unit(("gas", "density"), "Msun/kpc**3")
            sz_dens.set_zlim( field, 0.7e+0, 0.7e+4 )
            #sz_dens.annotate_scale()
            sz_dens.hide_axes()
            #sz.set_zlim( field, 1.0e-31, 1.0e-22 )
            sz_dens.zoom(2)
            sz_dens.set_axes_unit( 'Mpc' )
            #sz_dens.set_log("Phase", False)
            sz_dens.set_cmap( field, colormap )
            #sz_dens.set_zlim( "density", zmin=(5e-31, "g/cm**3"), zmax = (5e-27, "g/cm**3"))
            cbar = sz_dens.plots[field].cb
            cbar.ax.tick_params(labelsize=font_size)


            sz_dens.set_font({'size': time_size})
            sz_dens.annotate_scale(coeff=1, pos=(0.85, 0.05), corner='lower_right', text_args={'size': time_size})



            sz_dens.annotate_timestamp( time_unit='Gyr', redshift = True, corner='upper_right' )
            sz_dens.save('Data_%06d_%06d_Slice_%s_density.png'%(num, j, ax), mpl_kwargs={"dpi":dpi} )
            #sz_dens.save('Data_%06d_%06d_Slice_%s_phase.png'%(num, j, ax), mpl_kwargs={"dpi":dpi} )
            sz_dens.annotate_grids(min_level=4, edgecolors='w')
            sz_dens.save('Data_%06d_%06d_Slice_%s_density_grid.png'%(num, j, ax), mpl_kwargs={"dpi":dpi} )
            field = ("gamer", "Phase")
            sz_phas = yt.SlicePlot( ds, ax, ("gamer", "Phase"), center = center) #, center=[coordinate_x, coordinate_y, coordinate_z] )
            sz_phas.set_axes_unit( 'Mpc' )
            sz_phas.set_cmap( field, colormap )
            sz_phas.set_log( field, False )
            #sz_phas.annotate_scale()
            sz_phas.hide_axes()
            #sz.set_zlim( field, 1.0e-31, 1.0e-22 )
            sz_phas.zoom(2)
            cbar = sz_phas.plots[field].cb
            cbar.ax.tick_params(labelsize=font_size)


            sz_phas.set_font({'size': font_size})
            sz_phas.annotate_scale(corner='lower_right', text_args={'size': font_size})


            sz_phas.annotate_timestamp( time_unit='Gyr', corner='upper_right' )
            sz_phas.save('Data_%06d_%06d_Slice_%s_phase.png'%(num, j, ax), mpl_kwargs={"dpi":dpi} )
            sz_phas.annotate_grids(min_level=4, edgecolors='w')
            sz_phas.save('Data_%06d_%06d_Slice_%s_phase_grid.png'%(num, j, ax), mpl_kwargs={"dpi":dpi} )
            field = ("gas", "unwrapped_phase")
            sz_phas = yt.SlicePlot( ds, ax, field, center = center) #, center=[coordinate_x, coordinate_y, coordinate_z] )
            sz_phas.set_zlim( field, 0, 380 )
            sz_phas.set_axes_unit( 'Mpc' )
            sz_phas.set_cmap( field, colormap )
            sz_phas.set_log( field, False )
            #sz_phas.annotate_scale()
            sz_phas.hide_axes()
            #sz.set_zlim( field, 1.0e-31, 1.0e-22 )
            sz_phas.zoom(2)
    
            cbar = sz_phas.plots[field].cb
            cbar.ax.tick_params(labelsize=font_size)


            sz_phas.set_font({'size': time_size})
            sz_phas.annotate_scale(coeff=1, pos=(0.85, 0.05), corner='lower_right', text_args={'size': time_size})



            sz_phas.annotate_timestamp( time_unit='Gyr', redshift=True, corner='upper_right' )
            sz_phas.save('Data_%06d_%06d_Slice_%s_phase_wrapped.png'%(num, j, ax), mpl_kwargs={"dpi":dpi} )
            sz_phas.annotate_grids(min_level=4, edgecolors='w')
            sz_phas.save('Data_%06d_%06d_Slice_%s_phase_grid_wrapped.png'%(num, j, ax), mpl_kwargs={"dpi":dpi} )
