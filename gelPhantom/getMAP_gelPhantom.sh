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
# module load anaconda3/2020.2
module load anaconda3/5.3.0
# install cil+pylops when running it for the first time:
# cd /home/slan7/Projects/STBP/code
# conda activate base
# conda create --prefix=cil -c conda-forge -c intel -c ccpi -c astra-toolbox cil cil-astra
# conda activate ./cil
# conda install --channel conda-forge pylops
# activate cil
source activate /home/slan7/Projects/STBP/code/cil

# go to working directory
cd ~/Projects/STBP/code/gelPhantom

# run python script
python -u gelPhantom.py #> gelPhantom-MAP.log &
# sbatch --job-name=gelPhantom --output=gelPhantom.log getMAP_gelPhantom.sh