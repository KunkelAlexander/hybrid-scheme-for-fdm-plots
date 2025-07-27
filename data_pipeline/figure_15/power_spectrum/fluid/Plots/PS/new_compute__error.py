import argparse
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yt
import numpy as np 
from scipy import interpolate
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
idx =  0
ds = yt.load(  prefix+'/Data_%06d'%idx   )
z_ref = ds.current_redshift
a_ref = 1/(1+z_ref)
#current_time_z = (1.0/(current_time_a)) - 1.0
df_ref = pd.read_csv( prefix+'/PowerSpec_%06d'%idx , sep = '\s+' , header = 1, names=["k", "Power", "Stubus"])
print(df_ref)
P_ref = df_ref["Power"]
ds.close()

for idx in range( idx_start, idx_end+1, didx ):

    ds = yt.load(  prefix+'/Data_%06d'%idx   )
    current_time_z = ds.current_redshift
    print(current_time_z, redshifts[idx])
    #current_time_z = (1.0/(current_time_a)) - 1.0
    df = pd.read_csv( prefix+'/PowerSpec_%06d'%idx , sep = '\s+' , header = 1, names=["k", "Power", "Stubus"])
    
    #prefix = "/projectY/vivi235711/LSS_boxsize/m22=0.2_L=5.6_N=512_MaxLevel=6/"
    #df = pd.read_csv( prefix+'/PowerSpec_%06d'%idx , sep = '\s+' , header = 0 )
    #plt.plot( df['k'], df['Power'],'.', c="g", label="Pin-Yu" )
    a = 1/(1 + current_time_z)
    P_lin = (a/a_ref)**2 * P_ref


    fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex=True)

    ax1.plot(df["k"], df["Power"], label="Hybrid scheme")
    ax1.plot(df["k"], P_lin, label="Linear PT")
    ax1.legend()

    error = np.abs(df["Power"] - P_lin) / (P_lin)  
    ax2.plot(df["k"], error, label="Hybrid vs linear PT")

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_xlim(2*np.pi/10,20)
    ax1.set_ylim(1e-6, 1e2)

    ax1.set_ylabel(r'$P(k)$')

    #ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlim(2*np.pi/10, 20)
    #ax2.set_ylim(0.1, 100)
    ax2.set_ylim(-0.02, 0.5)
    ax2.set_xlabel('k')
    ax2.set_ylabel(r'$(P - P_{ref})/P_{ref}$')
    #ax2.set_ylabel(r'$\frac{P}{P_{ref}}$')
    ax2.legend()

    fig.suptitle('Fluid N = 128: z = %.2e'%current_time_z, fontsize=12)

    FileOut = 'fig_error_%06d' %idx+'.png'
    plt.savefig( FileOut)
    plt.close()
