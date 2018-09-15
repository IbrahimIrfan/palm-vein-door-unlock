from processing import processImage
from servo import *

servo = initGPIO()

while(True):
    input = raw_input("Ready.")
    if "run" == input:
        #take image
        #upload raw
        #process image
        #upload processed
        #call model
        #check if authenticated
        #actuate servo
        #send label
    else if "lock" == input:
        reset(sevo)
    else if "quit" == input:
        cleanup(servo)
        break
    else:
        print("Invalid command.")