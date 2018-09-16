import subprocess

from post import post_label, post_original, post_processed
from processing import process_image
from servo import actuate, cleanup, init_GPIO, reset

servo = init_GPIO()
img_shape = (224, 224)
model_path = "./data/palm_vein_model.h5"
raw_img_path = "pic.jpg"
processed_img_path = "thr.jpg"

while(True):
    input = raw_input("Ready.\n")
    if "run" == input:
        subprocess.call("./take_pic.sh")
        print("image taken")
        post_original()
        process_image(raw_img_path, processed_img_path, img_shape)
        res = post_processed()

        isAuthenticated = False
        if res == 't':
            isAuthenticated = True

        if(isAuthenticated):
            actuate(servo)

    elif "lock" == input:
        reset(servo)
    elif "quit" == input:
        cleanup(servo)
        break
    else:
        print("Invalid command.\n")
