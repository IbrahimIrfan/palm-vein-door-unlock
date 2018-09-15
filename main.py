from processing import processImage
from servo import *
from post import *
import subprocess

servo = initGPIO()

while(True):
    input = raw_input("Ready.\n")
    if "run" == input:
        subprocess.call("./takePic.sh")
        postOriginal()
        processImage("pic.jpg", "thr.jpg")
        postProcessed()
        #TODO: call model, return label
        #check if authenticated
        isAuthenticated = 
        postLabel(label, isAuthenticated)
        if(isAuthenticated):
            actuate(servo)
    else if "lock" == input:
        reset(servo)
    else if "quit" == input:
        cleanup(servo)
        break
    else:
        print("Invalid command.\n")