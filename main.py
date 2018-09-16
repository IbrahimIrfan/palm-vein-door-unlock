import subprocess

from keras.models import load_model

import cv2
from post import post_label, post_original, post_processed
from processing import process_image, load_img
from servo import actuate, cleanup, init_GPIO, reset

servo = init_GPIO()
model_path = "./data/palm_vein_model.h5"
raw_img_path = "pic.jpg"
processed_img_path = "thr.jpg"
anchor_img_path = "./data/processed/ayush_right_1.jpg"

while(True):
    input = raw_input("Ready.\n")
    if "run" == input:
        subprocess.call("./take_pic.sh")
        post_original()
        process_image(raw_img_path, processed_img_path)
        post_processed()

        input_img = load_img(processed_img_path, (224, 224))
        anchor_img = load_img(anchor_img_path, (224, 224))

        model = load_model(model_path)
        y_pred = model.predict([input_img, anchor_img])
        isAuthenticated = False
        if y_pred < 0.2:
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
