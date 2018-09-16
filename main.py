import subprocess

from keras.models import load_model

from processing import process_image
from post import post_label, post_original, post_processed
from servo import actuate, cleanup, init_GPIO, reset

servo = init_GPIO()
model_path = './data/palm_vein_model.h5'

while(True):
    input = raw_input("Ready.\n")
    if "run" == input:
        subprocess.call("./take_pic.sh")
        post_original()
        process_image("pic.jpg", "thr.jpg")
        post_processed()
        model = load_model(model_path)
        #TODO: call model, return label
        #check if authenticated
        isAuthenticated = True
        post_label(label, isAuthenticated)
        if(isAuthenticated):
            actuate(servo)
    elif "lock" == input:
        reset(servo)
    elif "quit" == input:
        cleanup(servo)
        break
    else:
        print("Invalid command.\n")
