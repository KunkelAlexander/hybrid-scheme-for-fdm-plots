import argparse
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yt
import numpy as np 

# load the command-line parameters
parser = argparse.ArgumentParser( description='Power Spectrum' )

parser.add_argument( '-i', action='store', required=False, type=str, dest='prefix',
                     help='path prefix [%(default)s]', default='../../' )
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
    print str(sys.argv[t]),
print( '' )
print( '-------------------------------------------------------------------\n' )


idx_start = args.idx_start
idx_end   = args.idx_end
didx      = args.didx
prefix    = args.prefix

# ts = yt.load( [ prefix+'/Data_%06d'%idx for idx in range(idx_start, idx_end+1, didx) ] )
redshifts = [99.0, 19.0, 9.99999987, 9.36241251, 8.77859223, 8.24264742, 7.74947079, 7.29461736, 6.87420394, 6.48482599, 6.12348847, 5.78754808, 5.47466476, 5.18276085, 4.9099865, 4.65469032, 4.41539425, 4.19077222]
for idx in range( idx_start, idx_end+1, didx ):

    prefix = "../../"
    ds = yt.load(  prefix+'/Data_%06d'%idx   )
    current_time_z = ds.current_redshift
    print(current_time_z, redshifts[idx])
    #current_time_z = (1.0/(current_time_a)) - 1.0
    df = pd.read_csv( prefix+'/PowerSpec_%06d'%idx , sep = '\s+' , header = 0 )
    plt.plot( df['k'], df['Power'],'.',c="r", label="Hybrid" )
    
    
    #prefix = "/projectY/vivi235711/LSS_boxsize/m22=0.2_L=5.6_N=512_MaxLevel=6/"
    #df = pd.read_csv( prefix+'/PowerSpec_%06d'%idx , sep = '\s+' , header = 0 )
    #plt.plot( df['k'], df['Power'],'.', c="g", label="Pin-Yu" )

    prefix = "/work1/kunkelalexander/oldAxionCAMB/10Box/z%s_matterpower.dat"%(str(redshifts[idx]))
    k, P = np.loadtxt( prefix, unpack = True )
    plt.plot(k, P,'.', c="k", label="axionCAMB" )

    plt.xscale( 'log' )
    plt.yscale( 'log' )
    
    plt.xlim(1,100)
    #plt.ylim(1e-10,1e2)
    plt.xlabel('k')
    plt.ylabel('p')
    plt.legend()
    plt.title('z = %.2e'%current_time_z, fontsize=12)

    FileOut = 'fig_powerspec_%06d' %idx+'.png'
    plt.savefig( FileOut)
    plt.close()

