#!/bin/bash
 
#SBATCH -N 1                        # number of compute nodes
#SBATCH -c 5                        # number of "tasks" (cores)
#SBATCH --mem=64G                   # GigaBytes of memory required (per node)

#SBATCH -p publicgpu                # Use gpu partition
#SBATCH -q wildfire                 # Run job under wildfire QOS queue
#SBATCH --gres=gpu:1                # Request two GPUs

#SBATCH -t 4-00:00                  # wall time (D-HH:MM)
##SBATCH -A slan7                   # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o %x.log                   # STDOUT (%j = JobId)
#SBATCH -e %x.err                   # STDERR (%j = JobId)
#SBATCH --mail-type=END,FAIL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=slan@asu.edu    # send-to address

# load environment
# module load python/3.7.1
# module load anaconda3/2020.2
# source activate /packages/7x/anaconda3/5.3.0/envs/pytorch-gpu-1.9.0
source /home/slan7/miniconda/bin/activate
conda activate pytorch

# go to working directory
cd ~/Projects/STBP/code/NSE

python -u ns_2d.py #> sol_NSE.log

# sbatch --job-name=sol --output=sol_NSE.log solve_NSE.sh