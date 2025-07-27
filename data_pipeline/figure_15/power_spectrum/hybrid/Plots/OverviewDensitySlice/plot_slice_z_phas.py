import argparse
import sys
import yt
import pandas as pd
import numpy as np
# load the command-line parameters
parser = argparse.ArgumentParser( description='Slice of mass phase' )

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

field       = 'Phase'
colormap    = 'algae'
center_mode = 'c'
dpi         = 150


yt.enable_parallelism()


ts = yt.DatasetSeries( [ prefix+'/Data_%06d'%idx for idx in range(idx_start, idx_end+1, didx) ] )

for ds in ts.piter():

   num = '%s'%ds
   num = int(num[9:11])

   for j, coordinates in enumerate(np.linspace(0, 4, 3)):
       for ax in ["x", "y", "z"]:
           if ax == "x":
		center = [coordinates, 0, 0]
	   elif ax == "y":
		center = [0, coordinates, 0]
	   elif ax == "z":
		center = [0, 0, coordinates]

	   sz_dens = yt.SlicePlot( ds, ax, ("gamer", "Phase"), center = center) #, center=[coordinate_x, coordinate_y, coordinate_z] )
	   sz_dens.set_axes_unit( 'Mpc' )
	   sz_dens.set_cmap( field, colormap )
	   sz_dens.set_log( field, False )
	   sz_dens.annotate_timestamp( time_unit='Gyr', corner='upper_right' )
	   sz_dens.save('Data_%06d_%06d_Slice_%s_phase.png'%(num, j, ax), mpl_kwargs={"dpi":dpi} )
	   sz_dens.annotate_grids(edgecolors='w')
	   sz_dens.save('Data_%06d_%06d_Slice_%s_phase_grid.png'%(num, j, ax), mpl_kwargs={"dpi":dpi} )
