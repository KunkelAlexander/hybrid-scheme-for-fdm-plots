import argparse
import sys
import yt
import numpy as np
# load the command-line parameters
parser = argparse.ArgumentParser( description='Projection of phase' )

parser.add_argument( '-i', action='store', required=False, type=str, dest='prefix',
                     help='path prefix [%(default)s]', default='../../' )
parser.add_argument( '-s', action='store', required=True,  type=int, dest='idx_start',
                     help='first data index' )
parser.add_argument( '-e', action='store', required=True,  type=int, dest='idx_end',
                     help='last data index' )
parser.add_argument( '-d', action='store', required=False, type=int, dest='didx',
                     help='delta data index [%(default)d]', default=1 )

args=parser.parse_args()


idx_start = args.idx_start
idx_end   = args.idx_end
didx      = args.didx
prefix    = args.prefix

field         = 'PhaseMod2'
colormap_dens = 'algae'
center_mode   = 'c'
dpi           = 150

def _phase(field, data):
  return np.arctan2(np.sin(data["gamer", "Phase"]), np.cos(data["gamer", "Phase"]))

yt.add_field(
name=("gamer", "PhaseMod2"),
function=_phase,
sampling_type="local",
units="",
)
 
yt.enable_parallelism()
ts = yt.DatasetSeries( [ prefix+'/Data_%06d'%idx for idx in range(idx_start, idx_end+1, didx) ] )

for ds in ts.piter():

   pz_dens = yt.ProjectionPlot( ds, 'x', field, center=center_mode )
   # pz_dens.zoom(2)
   pz_dens.set_axes_unit( 'code_length' )
   #pz_dens.set_zlim( field, 3.3e-5, 3.6e-5 )
   pz_dens.set_cmap( field, colormap_dens )
   # pz_dens.annotate_grids()
   pz_dens.annotate_timestamp( time_unit='Gyr', corner='upper_right' )
   #redshift_format='z = {redshift:.2f}'
   pz_dens.save( mpl_kwargs={"dpi":dpi} )
