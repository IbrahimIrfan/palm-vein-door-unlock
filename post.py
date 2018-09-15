import requests
import base64
url = "http://192.168.43.153:8000/"

def postOriginal():
    postImage("pic")

def postProcessed():
    postImage("thr")

def postImage(name):
    with open(name + ".jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        payload = {"type":"image", "img": encoded_string, "name":name}
        requests.post(url, data=payload)

def postLabel(label, isAuthenticated):
    payload = {"type":"label", "label": label, "auth": isAuthenticated}
    requests.post(url, data=payload)
