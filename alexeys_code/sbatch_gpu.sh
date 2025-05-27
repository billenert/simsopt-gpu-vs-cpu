#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --account=m4680
#SBATCH -J gpu_test
#SBATCH --ntasks-per-node=1

module load conda cray-hdf5-parallel cray-netcdf-hdf5parallel
conda activate gpu_orm_minimal
python -u trace_lost_trajectories_gpu.py | tee output_gpu.txt
