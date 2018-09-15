#!/bin/bash
#Collect data to train palm verification system
#Pictures are in the format name_leftORright_pic#
while :
do
    echo 'Enter name (lowercase)'
    read name
    #do right training data collection
    for i in {0..4}
    do
    	fileNameRight=${name}_right_${i}.jpg
    	echo $fileNameRight
    	raspistill -h 600 -w 600 -roi 0.45,0.57,0.18,0.18 -o "raw/$fileNameRight"
    done

    echo "switch to left hand, press enter"
    read garbage

    #do left training data collection
    for i in {0..4}
    do
    	fileNameLeft=${name}_left_${i}.jpg
    	echo $fileNameLeft
    	raspistill -h 600 -w 600 -roi 0.45,0.57,0.18,0.18 -o "raw/$fileNameLeft"
    done
done
