import requests
import base64
url = "http://192.168.43.153:8000/"

with open("pic.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
    payload = {'img': encoded_string}
    r = requests.post(url, data=payload)
    print r
