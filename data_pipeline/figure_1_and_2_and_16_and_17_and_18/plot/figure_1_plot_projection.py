import argparse
import sys
import yt

# global parameter settings
field         = ('gas', 'density')
axes          = ["x", "y", "z"]                            # directions
lv            = 10                                         # maximum level for sampling AMR grid
dpi           = 300
colormap_dens = 'magma'
center_mode   = 'c'

# load the command-line parameters
parser = argparse.ArgumentParser( description='Projection of mass density' )

parser.add_argument( '-i', action='store', required=False, type=str, dest='prefix',
                     help='path prefix [%(default)s]', default='./' )
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


idx_start = args.idx_start
idx_end   = args.idx_end
didx      = args.didx
prefix    = args.prefix
font_size             = 42


yt.enable_parallelism()
ts = yt.DatasetSeries( [ prefix+'/Data_%06d'%idx for idx in range(idx_start, idx_end+1, didx) ] )

for ds in ts.piter():
   num = '%s'%ds
   num = int(num[5:11])
   for ax in axes:
      ad           = ds.all_data()
      ad.max_level = lv



      pz_dens = yt.ProjectionPlot( ds, ax, field, center=center_mode, data_source=ad )
      pz_dens.hide_axes()
      pz_dens.colorbar_label = "Projected Density"
      # Set font sizes
      pz_dens.set_font({'size': font_size})
      pz_dens.annotate_scale(corner='lower_right', text_args={'size': font_size})


      # Convert the units of the density field to Msun/kpc**3
      pz_dens.set_unit(("gas", "density"), "Msun/kpc**3")

      pz_dens.set_zlim( field, 1.0e+5, 2.0e+8 )
      pz_dens.set_cmap( field, colormap_dens )
      pz_dens.annotate_timestamp( time_unit='Gyr', redshift=True, corner='upper_right' )
      
      # Modify the colorbar font size directly
      cbar = pz_dens.plots[("gas", "density")].cb
      cbar.ax.tick_params(labelsize=font_size)

      pz_dens.save('Data_%06d_Proj_%s_%s.png'%(num, ax, field[1]), mpl_kwargs={"dpi":dpi} )

      pz_dens.annotate_grids(min_level=4, max_level=4, edgecolors="white")
      pz_dens.save('Data_%06d_Proj_%s_%s_grid.png'%(num, ax, field[1]), mpl_kwargs={"dpi":dpi} )
