#!/bin/bash
#Collect data to train palm verification system
#Pictures are in the format name_leftORright_pic#
while :
do
    echo 'Enter name (lowercase)'
	read name
    #do right data collection
    for i in {1..10}
    do
    fileNameRight = "%name_right_%i.jpg"
    raspistill -h 600 -w 600 -roi 0.45,0.57,0.18,0.18 -o %fileNameRight
    done
    #do left data collection
    for i in {1..10}
    do
    fileNameLeft = "%name_left_%i.jpg"
    raspistill -h 600 -w 600 -roi 0.45,0.57,0.18,0.18 -o %fileNameLeft
    done
done