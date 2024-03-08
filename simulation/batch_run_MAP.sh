#!/bin/bash

# run python script
if [ $# -eq 0 ]; then
	bas_NO=0
	wav_NO=1
	ker_NO=1
	q=1
	whiten=1
	NCG=0
elif [ $# -eq 1 ]; then
	bas_NO="$1"
	wav_NO=1
	ker_NO=1
	q=1
	whiten=1
	NCG=0
elif [ $# -eq 2 ]; then
	bas_NO="$1"
	wav_NO="$2"
	ker_NO=1
	q=1
	whiten=1
	NCG=0
elif [ $# -eq 3 ]; then
	bas_NO="$1"
	wav_NO="$2"
	ker_NO="$3"
	q=1
	whiten=1
	NCG=0
elif [ $# -eq 4 ]; then
	bas_NO="$1"
	wav_NO="$2"
	ker_NO="$3"
	q="$4"
	whiten=1
	NCG=0
elif [ $# -eq 5 ]; then
	bas_NO="$1"
	wav_NO="$2"
	ker_NO="$3"
	q="$4"
	whiten="$5"
	NCG=0
elif [ $# -eq 6 ]; then
	bas_NO="$1"
	wav_NO="$2"
	ker_NO="$3"
	q="$4"
	whiten="$5"
	NCG="$6"
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

for q in {0..2}
do
	if [ ${q} -eq 0 ]; then
		mdl_name='iidT'
	elif [ ${q} -eq 1 ]; then
		mdl_name='STBP'
	elif [ ${q} -eq 2 ]; then
		mdl_name='STGP'
	else
		echo "Wrong args!"
		exit 0
	fi
	for n_x in 16 32 128 256
	do
		for n_t in 10 20 50 100
		do
			sbatch --job-name=${mdl_name}-I${n_x}-J${n_t} --output=MAP-${mdl_name}-I${n_x}-J${n_t}.log run_MAP.sh ${n_x} ${n_t} ${bas_NO} ${wav_NO} ${ker_NO} ${q} ${whiten} ${NCG}
			echo "Job MAP-${mdl_name}-I${n_x}-J${n_t} submitted."
		done
	done
done
