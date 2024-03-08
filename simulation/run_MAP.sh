#!/bin/bash
 
#SBATCH -N 1                        # number of compute nodes
#SBATCH -c 5                        # number of "tasks" (cores)
#SBATCH --mem=64G                   # GigaBytes of memory required (per node)

#SBATCH -p parallel                 # partition 
#SBATCH -q normal                   # QOS

#SBATCH -t 0-6:00                  # wall time (D-HH:MM)
##SBATCH -A slan7                   # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o %x.log                   # STDOUT (%j = JobId)
#SBATCH -e %x.err                   # STDERR (%j = JobId)
#SBATCH --mail-type=FAIL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=slan@asu.edu    # send-to address

# load environment
# module load python/3.7.1
module load anaconda3/2020.2

# go to working directory
cd ~/Projects/STBP/code/simulation

# run python script
if [ $# -eq 0 ]; then
	n_x=16
	n_t=10
	bas_NO=0
	wav_NO=1
	ker_NO=1
	q=1
	whiten=0
	NCG=0
elif [ $# -eq 1 ]; then
	n_x="$1"
	n_t=10
	bas_NO=0
	wav_NO=1
	ker_NO=1
	q=1
	whiten=0
	NCG=0
elif [ $# -eq 2 ]; then
	n_x="$1"
	n_t="$2"
	bas_NO=0
	wav_NO=1
	ker_NO=1
	q=1
	whiten=0
	NCG=0
elif [ $# -eq 3 ]; then
	n_x="$1"
	n_t="$2"
	bas_NO="$3"
	wav_NO=1
	ker_NO=1
	q=1
	whiten=0
	NCG=0
elif [ $# -eq 4 ]; then
	n_x="$1"
	n_t="$2"
	bas_NO="$3"
	wav_NO="$4"
	ker_NO=1
	q=1
	whiten=0
	NCG=0
elif [ $# -eq 5 ]; then
	n_x="$1"
	n_t="$2"
	bas_NO="$3"
	wav_NO="$4"
	ker_NO="$5"
	q=1
	whiten=0
	NCG=0
elif [ $# -eq 6 ]; then
	n_x="$1"
	n_t="$2"
	bas_NO="$3"
	wav_NO="$4"
	ker_NO="$5"
	q="$6"
	whiten=0
	NCG=0
elif [ $# -eq 7 ]; then
	n_x="$1"
	n_t="$2"
	bas_NO="$3"
	wav_NO="$4"
	ker_NO="$5"
	q="$6"
	whiten="$7"
	NCG=0
elif [ $# -eq 8 ]; then
	n_x="$1"
	n_t="$2"
	bas_NO="$3"
	wav_NO="$4"
	ker_NO="$5"
	q="$6"
	whiten="$7"
	NCG="$8"
fi

if [ ${bas_NO} -eq 0 ]; then
	bas_name='Fourier'
elif [ ${bas_NO} -eq 1 ]; then
	bas_name='wavelet'
else
	echo "Wrong args!"
	exit 0
fi

if [ ${wav_NO} -eq 0 ]; then
	wav_name='Harr'
elif [ ${wav_NO} -eq 1 ]; then
	wav_name='Shannon'
elif [ ${wav_NO} -eq 2 ]; then
	wav_name='Meyer'
elif [ ${wav_NO} -eq 3 ]; then
	wav_name='MexHat'
elif [ ${wav_NO} -eq 4 ]; then
	wav_name='Poisson'
else
	echo "Wrong args!"
	exit 0
fi

if [ ${ker_NO} -eq 0 ]; then
	ker_name='powexp'
elif [ ${ker_NO} -eq 1 ]; then
	ker_name='matern'
else
	echo "Wrong args!"
	exit 0
fi

python -u run_simulation_MAP.py ${n_x} ${n_t} ${bas_NO} ${wav_NO} ${ker_NO} ${q} ${whiten} ${NCG}
# sbatch --job-name=${bas_name}-${ker_name} --output=MAP-${bas_name}-${ker_name}.log run_MAP.sh