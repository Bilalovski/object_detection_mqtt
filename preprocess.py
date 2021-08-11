import json

import paho.mqtt.client as paho
import numpy as np
from PIL import Image
import pickle
import time
from pydust import core
import sys

RCNN = 1
MOBINET = 2

published = False


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


def receive(arg):
    image = pickle.loads(arg)
    param = sys.argv
    choice = 1
    if len(param) > 2:
        choice = param [1]
        model_path = param[2]
        label_paths = param[3]
    if len(param) == 1:
        choice = 1
    print(choice)
    print("received image")
    if choice == 1:
        print("rcnn")
        result = preprocess_RCNN(image)
        data = {'choice': choice, 'data': result.tolist()}
        payload = json.dumps(data)
        client.publish("preprocess_out", payload, qos=0)

    if choice == 2:
        print("mobinet")
        result = preprocess_mobinet(image)
        data = {'choice': choice, 'data': result.tolist()}
        payload = json.dumps(data)
        client.publish("preprocess_out", payload, qos=0)


    while not published:
        pass
    print("published")


def on_connect(mqtt_client, obj, flags, rc):
    if rc == 0:
        print("connected")
    else:
        print("connection refused")


def on_publish(client, userdata, mid):
    print("published data")
    global published
    published = True


broker = "broker.mqttdashboard.com"
client = paho.Client("preprocessor")
client.on_connect = on_connect
client.on_publish = on_publish
client.connect(broker)
client.loop_start()
dust = core.Core("OD_sub", "./modules")

# start a background thread responsible for tasks that shouls always be running in the same thread
dust.cycle_forever()
# load the core, this includes reading the libraries in the modules directory to check addons and transports are available
dust.setup()
# set the path to the configuration file
dust.set_configuration_file("configuration.json")
# connects all channels
dust.connect()
time.sleep(1)
# add a message listener on the subscribe-tcp channel. The callback function takes a bytes-like object as argument containing the payload of the message
dust.register_listener("OD_image", receive)
# dust.register_listener("subscribe-mqtt", lambda payload: print("Received payload with %d bytes" % len(payload)))

while True:
    time.sleep(1)
