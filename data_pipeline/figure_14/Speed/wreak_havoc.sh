#!/bin/bash

# Base directory for the gamer source code and for moving executables
GAMER_SRC_DIR="/work1/kunkelalexander/gamer-fork/src"
BASE_RUN_DIR="/work1/kunkelalexander/hybrid_scheme_paper/speed/GaussianWavePacket"

# Configuration settings for each type of compilation
declare -A settings
settings[fd2_cpu]="--elbdm_scheme=ELBDM_WAVE --wave_scheme=WAVE_FD --laplacian_four=false --gpu=false"
settings[fd2_gpu]="--elbdm_scheme=ELBDM_WAVE --wave_scheme=WAVE_FD --laplacian_four=false --gpu=true"
settings[fd4_cpu]="--elbdm_scheme=ELBDM_WAVE --wave_scheme=WAVE_FD --laplacian_four=true --gpu=false"
settings[fd4_gpu]="--elbdm_scheme=ELBDM_WAVE --wave_scheme=WAVE_FD --laplacian_four=true --gpu=true"
settings[fluid_cpu]="--elbdm_scheme=ELBDM_HYBRID --wave_scheme=WAVE_FD --laplacian_four=true --gpu=false"
settings[fluid_gpu]="--elbdm_scheme=ELBDM_HYBRID --wave_scheme=WAVE_FD --laplacian_four=true --gpu=true"
settings[gramfe_fft_cpu]="--elbdm_scheme=ELBDM_WAVE --wave_scheme=WAVE_GRAMFE --gramfe_scheme=GRAMFE_FFT --gpu=false"
settings[gramfe_fft_gpu]="--elbdm_scheme=ELBDM_WAVE --wave_scheme=WAVE_GRAMFE --gramfe_scheme=GRAMFE_FFT --gpu=true"
settings[gramfe_matmul_cpu]="--elbdm_scheme=ELBDM_WAVE --wave_scheme=WAVE_GRAMFE --gramfe_scheme=GRAMFE_MATMUL --gpu=false"
settings[gramfe_matmul_gpu]="--elbdm_scheme=ELBDM_WAVE --wave_scheme=WAVE_GRAMFE --gramfe_scheme=GRAMFE_MATMUL --gpu=true"

# Loop through each setting
for folder in "${!settings[@]}"; do
    echo "Processing $folder with settings: ${settings[$folder]}"
    
    # Navigate to the gamer source directory and generate the Makefile with specific settings
    cd $GAMER_SRC_DIR
    python3 configure.py --machine=spock_intel --mpi=true --hdf5=true --fftw=FFTW3 --gpu_arch=AMPERE \
                       --model=ELBDM ${settings[$folder]} --gravity=false --comoving=false --gsl=true \
                       --spectral_interpolation=true --double=false --timing_solver=false
    
    # Compile the executable
    make clean && make -j 16
    
    # Move the executable to the appropriate run folder and submit the job
    if [ -f gamer ]; then
        mv gamer "$BASE_RUN_DIR/$folder/"
        cd "$BASE_RUN_DIR/$folder"
	sh clean.sh
        qsub submit_spock.job
    else
        echo "Compilation failed or gamer executable not found for $folder"
    fi
done

echo "All processes completed."

