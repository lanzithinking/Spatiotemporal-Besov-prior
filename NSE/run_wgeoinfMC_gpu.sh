#!/bin/bash
 
#SBATCH -N 1                        # number of compute nodes
#SBATCH -c 5                        # number of "tasks" (cores)
#SBATCH --mem=128G                   # GigaBytes of memory required (per node)

#SBATCH -p gpu                # partition 
#SBATCH -q wildfire                 # Run job under wildfire QOS queue
#SBATCH --gres=gpu:2                # Request two GPUs

#SBATCH -t 2-00:00                  # wall time (D-HH:MM)
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

# run python script
if [ $# -eq 0 ]; then
	alg_NO=0
	seed_NO=2022
	q=1
elif [ $# -eq 1 ]; then
	alg_NO="$1"
	seed_NO=2022
	q=1
elif [ $# -eq 2 ]; then
	alg_NO="$1"
	seed_NO="$2"
	q=1
elif [ $# -eq 3 ]; then
	alg_NO="$1"
	seed_NO="$2"
	q="$3"
fi


python -u run_NSE_wgeoinfMC.py ${alg_NO} ${seed_NO} ${q} #> winfmMALA_${q}.log

# sbatch --job-name=winfmMALA --output=winfmMALA.log run_wgeoinfMC_gpu.sh