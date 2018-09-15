#!/bin/bash
#Collect data to train palm verification system
#Pictures are in the format name_leftORright_pic#
while :
do
    echo 'Enter name (lowercase)'
	read name
    #do right training data collection
    for i in {0..5}
    do
    fileNameRight = "%name_right_%i.jpg"
    raspistill -h 600 -w 600 -roi 0.45,0.57,0.18,0.18 -o "raw/%fileNameRight"
    python -c"from ../processing import processing; processing.processImage("raw/%fileNameRight", "processed/%fileNameRight")"
    done
done