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

# go to working directory
cd ~/Projects/STBP/code/emoji

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


python -u run_emoji_wpcn.py ${seed_NO} ${q} #> wpcn_${q}.log

# sbatch --job-name=wpCN --output=wpCN.log run_wpcn.sh