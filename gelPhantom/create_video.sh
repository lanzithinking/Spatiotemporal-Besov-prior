#!/bin/bash

if [ $# -eq 0 ]; then
	src='.'
	dst='.'
	fname='video'
elif [ $# -eq 1 ]; then
	src="$1"
	dst='.'
	fname='video'
elif [ $# -eq 2 ]; then
	src="$1"
	dst="$2"
	fname='video'
elif [ $# -eq 3 ]; then
	src="$1"
	dst="$2"
	fname="$3"
fi

../util/ffmpeg -framerate 3 -i ${src}/${fname}_%02d.png -vcodec libx264 -s 640x480 -pix_fmt yuv420p ${dst}/${fname}.mp4

# ./create_video.sh ./reconstruction/MAP ./reconstruction/MAP gelPhantom