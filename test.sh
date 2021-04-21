echo "Started converting the videos to images that we will feed to the network"
python video.py
python produce_predictions.py
python generateVideo.py
