#!/bin/bash

sudo apt-get update
sudo apt-get install -y python3-pip ffmpeg

pip install -r requirements.txt

mkdir -p logs

echo "Installation complete!"
