#!/bin/bash -l
#SBATCH --job-name="get_nc"
#SBATCH --account="msrad"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=timo.schmid@usys.ethz.ch
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

# source /store/msrad/utils/miniconda-tschmid/bin/activate
# conda activate msrad
python /users/tschmid/scClimCSCS/nc_from_metranet.py
#Note: less than 30min for one year
