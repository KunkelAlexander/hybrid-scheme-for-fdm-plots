import argparse
import sys
import yt

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

args=parser.parse_args()

# take note
print( '\nCommand-line arguments:' )
print( '-------------------------------------------------------------------' )
for t in range( len(sys.argv) ):
   print(str(sys.argv[t]))
print( '' )
print( '-------------------------------------------------------------------\n' )


idx_start   = args.idx_start
idx_end     = args.idx_end
didx        = args.didx
prefix      = args.prefix

field       = 'density'
colormap    = 'algae'
center_mode = 'c'
dpi         = 150


yt.enable_parallelism()


ts = yt.DatasetSeries( [ prefix+'/Data_%06d'%idx for idx in range(idx_start, idx_end+1, didx) ] )

for ds in ts.piter():

   sz_dens = yt.SlicePlot( ds, 'z', field, center=[2.4, 2.4, 0.9] )
   sz_dens.zoom(5)
   sz_dens.set_axes_unit( 'code_length' )
   # sz_dens.set_zlim( field, 1e-31, 1.0e-26 )
   sz_dens.set_zlim( field, 1e-31, 1.0e-26 )
   sz_dens.set_cmap( field, colormap )
   sz_dens.annotate_timestamp( time_unit='Gyr', corner='upper_right' )
   sz_dens.annotate_grids()
   # sz_dens.annotate_cell_edges(line_width=0.001, alpha=0.7)
   sz_dens.save( mpl_kwargs={"dpi":dpi} )
