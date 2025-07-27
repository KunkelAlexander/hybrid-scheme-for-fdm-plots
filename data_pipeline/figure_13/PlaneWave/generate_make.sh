# This script should run in the same directory as configure.py

PYTHON=python3

${PYTHON} configure.py --machine=eureka_intel --mpi=true --hdf5=true --fftw=FFTW3 --gpu=false --gpu_arch=TURING \
                       --model=ELBDM --elbdm_scheme=ELBDM_WAVE --wave_scheme=WAVE_FD --laplacian_four=true --conserve_mass=false --gramfe_scheme=GRAMFE_FFT \
                       --gravity=false --comoving=false --gsl=true --spectral_interpolation=true --double=true --passive=1
