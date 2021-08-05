import json

import paho.mqtt.client as paho
import numpy as np
from PIL import Image
RCNN = 1
MOBINET = 2

published=False

def on_connect(mqtt_client, obj, flags, rc):
    if rc==0:
        print("connected")
    else:
        print("connection refused")

def on_publish(client, userdata, mid):
    print("published data")
    global published
    published=True

image = Image.open("demo.jpg")
broker = "127.0.0.1"
client = paho.Client("preprocessor")
client.on_connect=on_connect
client.on_publish=on_publish
client.connect(broker)
choice = 1
client.loop_start()
def preprocess_RCNN(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)
    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image
    return image


def preprocess_mobinet(img):
    img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
    return img_data


if choice == RCNN:
    result = preprocess_RCNN(image)
    data = {'choice': choice,'data': result.tolist()}
    payload = json.dumps(data)
    client.publish("preprocess_out", payload, qos=0)

if choice == MOBINET:
    result = preprocess_mobinet(image)
    data = {'choice': choice, 'data': result.tolist()}
    payload = json.dumps(data)
    client.publish("preprocess_out", payload, qos=0)

while not published:
    pass