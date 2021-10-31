import requests
import json

def get_detection_result(img_data):
    res = requests.post("http://{}:{}/ai/detection_predict".format("127.0.0.1", "8888"), data=img_data)
    print(res.text)

if __name__ == '__main__':
    with open("Test_Pic/fimg_0.jpg", "rb") as f:
        get_detection_result(f.read())
