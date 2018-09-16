import subprocess

import numpy as np
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.applications import mobilenet

import cv2
from post import post_label, post_original, post_processed
from processing import load_img, process_image
from servo import actuate, cleanup, init_GPIO, reset

servo = init_GPIO()
model_path = "./data/palm_vein_model.h5"
raw_img_path = "pic.jpg"
processed_img_path = "thr.jpg"
anchor_img_path = "./data/processed/ayush_right_1.jpg"
classes = ['angad_left', 'angad_right', 'anushka_left', 'anushka_right', 'ayush_left', 'ayush_right', 
            'cindy_left', 'cindy_right', 'david_left', 'david_right', 'edwin_left', 'edwin_right', 
            'ibrahim_left', 'ibrahim_right', 'jason_left', 'jason_right', 'jun_left', 'jun_right', 
            'justin_left', 'justin_right', 'nick_left', 'nick_right', 'samir_left', 'samir_right', 
            'thomas_left', 'thomas_right', 'will_left', 'will_right']

while(True):
    input = raw_input("Ready.\n")
    if "run" == input:
        subprocess.call("./take_pic.sh")
        post_original()
        process_image(raw_img_path, processed_img_path)
        post_processed()

        input_img = load_img(processed_img_path, (224, 224))
        
        with CustomObjectScope({'relu6': mobilenet.relu6,'DepthwiseConv2D': mobilenet.DepthwiseConv2D}):
            model = load_model(model_path)
            y_pred = model.predict(input_img)
            max_index = np.argmax(y_pred[0])
            label = classes[max_index]

        isAuthenticated = False
        if label == 'ayush_right' and y_pred[0][max_index] > 0.2:
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
