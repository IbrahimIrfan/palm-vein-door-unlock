import requests
import base64
url = "http://localhost:8000/"

with open("image.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
    payload = {'img': encoded_string}
    requests.post(url, data=payload)
