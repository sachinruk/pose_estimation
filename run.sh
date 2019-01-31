TIME=5 # length of video to extract in seconds
INPUT='dancing.mp4'
OUTPUT='output3.avi'
bash getModels.sh # get DL models
ffmpeg -ss 0 -i $INPUT -t $TIME audio.wav
python demo.py --input $INPUT --t $TIME --output tmp.avi
ffmpeg -i tmp.avi -i audio.wav -codec copy -shortest $OUTPUT
rm tmp.avi audio.wav #clean up extra files