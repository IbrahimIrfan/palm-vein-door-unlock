import requests
import base64
url = "http://192.168.43.97:8000/"
#url = "http://localhost:8000/"

def post_original():
    post_image("pic", "raw")

def post_processed():
    r = post_image("thr", "processed")
    return r

def post_image(name, extension):
    with open(name + ".jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        payload = {"type":"image", "img": encoded_string, "name":name}
        return requests.post(url + extension, data=payload)
        

def post_label(label, isAuthenticated):
    payload = {"type":"label", "label": label, "auth": isAuthenticated}
    requests.post(url, data=payload)
