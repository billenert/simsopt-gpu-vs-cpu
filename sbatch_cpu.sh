#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=m4680
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wcf2115@columbia.edu
#SBATCH -J cpu_test
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=output.txt

module load conda cray-hdf5-parallel cray-netcdf-hdf5parallel
conda activate gpu_orm_minimal
srun --mpi=pmix_v4 --cpu-bind=cores python -u tracing_cpu.py
