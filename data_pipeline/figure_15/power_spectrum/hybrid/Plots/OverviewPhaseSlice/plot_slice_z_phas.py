import argparse
import sys
import yt
import pandas as pd
import numpy as np

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

# take note
print( '\nCommand-line arguments:' )
print( '-------------------------------------------------------------------' )
for t in range( len(sys.argv) ):
   print(sys.argv[t], end = ' '),
print( '' )
print( '-------------------------------------------------------------------' )


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
   
   sz = yt.SlicePlot( ds, 'z', field)
   sz.set_axes_unit( 'Mpc' )
   sz.set_log(("gamer", "Phase"), False)
   sz.set_cmap( field, colormap )
   sz.annotate_timestamp( time_unit='Gyr', corner='upper_right' )
   sz.annotate_grids(edgecolors='w')
   # a = 1/(1+df['time'][num])
   # # print(a)
   # print(df['halo_radius'][num]/a)
   # sz_dens.annotate_sphere([coordinate_x, coordinate_y, coordinate_z], radius=(df['halo_radius'][num]/a, "kpc"))
   # # sz_dens.annotate_cell_edges(line_width=0.001, alpha=0.7)
   # sz_dens.save( mpl_kwargs={"dpi":dpi} )
   sz.save('Data_%06d_Slice_z_phase.png'%(num), mpl_kwargs={"dpi":dpi} )
   # sz_dens.zoom(20)
   # # sz_dens.annotate_grids()
   # sz_dens.set_zlim( field, 1e-28, 1.0e-21 )
   # sz_dens.save('Data_%06d_Slice_z_density_soliton.png'%(num), mpl_kwargs={"dpi":dpi} )
