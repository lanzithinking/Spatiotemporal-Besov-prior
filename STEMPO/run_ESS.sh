#!/bin/bash
 
#SBATCH -N 1                        # number of compute nodes
#SBATCH -c 5                        # number of "tasks" (cores)
#SBATCH --mem=64G                   # GigaBytes of memory required (per node)

#SBATCH -p parallel                 # partition 
#SBATCH -q normal                   # QOS

#SBATCH -t 1-12:00                  # wall time (D-HH:MM)
##SBATCH -A slan7                   # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o %x.log                   # STDOUT (%j = JobId)
#SBATCH -e %x.err                   # STDERR (%j = JobId)
#SBATCH --mail-type=END,FAIL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=slan@asu.edu    # send-to address

# load environment
# module load python/3.7.1
module load anaconda3/2020.2
# install astra+pylops when running it for the first time:
# cd /home/slan7/Projects/STBP/code
# conda activate base
# conda create --prefix=astra -c astra-toolbox astra-toolbox
# conda activate ./astra
# conda install --channel conda-forge pylops
# activate astra
source activate /home/slan7/Projects/STBP/code/astra

# go to working directory
cd ~/Projects/STBP/code/STEMPO

# run python script
 if [ $# -eq 0 ]; then
	seed_NO=2022
	q=1
elif [ $# -eq 1 ]; then
	seed_NO="$1"
	q=1
elif [ $# -eq 2 ]; then
	seed_NO="$1"
	q="$2"
fi


python -u run_stempo_ESS.py ${seed_NO} ${q} #> ESS_${q}.log

# sbatch --job-name=ESS --output=ESS.log run_ESS.sh