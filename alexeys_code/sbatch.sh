#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=m4680
#SBATCH -J cpu_test
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

module load conda cray-hdf5-parallel cray-netcdf-hdf5parallel
conda activate gpu_orm_minimal
srun --mpi=pmix_v4 --cpu-bind=cores python -u trace_lost_trajectories.py -m 0 | tee output.txt
