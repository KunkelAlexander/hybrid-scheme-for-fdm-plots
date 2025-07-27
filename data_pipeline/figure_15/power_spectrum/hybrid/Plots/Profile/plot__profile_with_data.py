import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys
import math
import yt

# load the command-line parameters
parser = argparse.ArgumentParser( description='Profile' )

parser.add_argument( '-p', action='store', required=False, type=str, dest='prefix',
                     help='path prefix [%(default)s]', default='../' )
parser.add_argument( '-s', action='store', required=True,  type=int, dest='idx_start',
                     help='first data index' )
parser.add_argument( '-e', action='store', required=True,  type=int, dest='idx_end',
                     help='last data index' )
parser.add_argument( '-d', action='store', required=False, type=int, dest='didx',
                     help='delta data index [%(default)d]', default=1 )

args=parser.parse_args()


idx_start   = args.idx_start
idx_end     = args.idx_end
didx        = args.didx
prefix      = args.prefix

halo = 1
# read Halo_Parameter
df_Halo_Parameter = pd.read_csv( 'Halo_Parameter' , sep = '\s+' , header = 0 , index_col='#')

# get background_density_0
ds = yt.load('../'+prefix+'Data_%06d'%idx_start)
omega_M0         = ds.omega_matter
hubble0          = ds.hubble_constant*100/1000  #km/(s*kpc)
newton_G         = 4.3*10**-6     #(kpc*km^2)/(s^2*Msun)
background_density_0 = (3*hubble0**2*omega_M0)/(8*math.pi*newton_G)

error = np.zeros((3,idx_end+1-idx_start))

for idx in range(idx_start, idx_end+1, didx):

    # read data
    radius, density = np.loadtxt('prof/Data_%06d_profile_data'%(idx) , skiprows=1, unpack=True)
    dens = density/background_density_0

    particle_mass  = df_Halo_Parameter['mass'][idx]
    current_time_z = df_Halo_Parameter['time'][idx]
    current_time_a = np.array(1/(current_time_z+1))
    core_radius_1  = np.array(df_Halo_Parameter['core_radius_1'][idx]/current_time_a)
    core_radius_2  = np.array(df_Halo_Parameter['core_radius_2'][idx]/current_time_a)

    print(df_Halo_Parameter["core_radius_1"][49])

    def soliton(x, core_radius):   
        return ((1.9*((current_time_a)**-1)*(particle_mass/10**-23)**-2*((core_radius)**-4))/((1 + 9.1*10**-2*(x/(core_radius))**2)**8))*10**9/background_density_0

    def r1(x):
        return x**-1*5e6

    def r3(x):
        return x**-3*7e7

    x = np.logspace(-1, 3, num=50)
    anal = soliton(x, core_radius_1)
    anal2 = soliton(x, core_radius_2)
        # calculate data - anal
    sum = 0
    sum2 = 0
    dens_error = 0
    dens_error2 = 0
    for i in range(len(radius)):
        if radius[i] < core_radius_1:
            dens_error += (dens[i] - soliton(radius[i], core_radius_1))/soliton(radius[i], core_radius_1)
            sum+=1
        if radius[i] < core_radius_2:
            dens_error2 += (dens[i] - soliton(radius[i], core_radius_2))/soliton(radius[i], core_radius_2)
            sum2+=1
    
    # dens_error =  dens_error/soliton(0, core_radius_1)/sum
    dens_error =  dens_error/sum
    dens_error2 =  dens_error2/sum2

    error[0,idx-idx_start] = current_time_a
    error[1,idx-idx_start] = dens_error
    error[2,idx-idx_start] = dens_error2

    ref1 = r1(x)
    ref3 = r3(x)

    # plt.plot( radius, dens, '.', label='Data_%02d'%idx,color=colorVal)
    plt.plot( radius, dens, 'bo', label='simulation')
    plt.plot( x, anal, label='analytical' )
    # plt.plot( x, anal, '--', label='analytical half_dens' )

    # plt.plot( x, ref1, '--' , label='r^-1' )    
    # plt.plot( x, ref3, '--' , label='r^-3' )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-1,1e3)
    plt.ylim(1e0,1e8)
    plt.ylabel('$\\rho(r)/\\rho_{m0}$')
    plt.xlabel('radius(kpc/a)')
    plt.annotate('error = %.2e'%dens_error, xy = (2e-1,10) )
    plt.legend(loc = 'upper right', fontsize=12)
    plt.title('z = %.2e core radius = %.2f kpc/a'%(current_time_z, core_radius_1), fontsize=12) 
    # plt.title('density profile', fontsize=12) 

    FileOut = 'fig_profile_density_%02d' %idx+'.png'
    plt.savefig( FileOut)
    plt.close()
    
# FileOut = 'fig_profile_density'+'.png'
# plt.savefig( FileOut)
# plt.plot(error[0] ,error[1]*100, label = 'by soliton function')
# plt.plot(error[0] ,error[2]*100, label = 'by 0.5 max density')
# plt.ylabel('error(%)')
# plt.xlabel('a')
# plt.legend()
# plt.title('density error in one r_c', fontsize=12) 
# plt.savefig( 'error.png')
