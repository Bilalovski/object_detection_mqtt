import json
import time

import paho.mqtt.client as paho
from matplotlib import pyplot as plt, patches
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from pydust import core

disp=False


def send_dust(payload):
    dust = core.Core("OD_pub", "./modules")

    # start a background thread responsible for tasks that shouls always be running in the same thread
    dust.cycle_forever()

    # load the core, this includes reading the libraries in the modules directory to check addons and transports are available
    dust.setup()

    # set the path to the configuration file
    dust.parse_configuration_file("configuration.json")

    # connects all channels
    dust.connect()
    time.sleep(1)
    # declare a bytes-like payload object
    # publishes the payload to the given channel (as defined by the configuration file)
    dust.publish("pub_postProcess_OD", payload)
    time.sleep(1)

    # disconnects all channels and flushes the addon stack and transport.
    dust.disconnect()

    # stops the background thread started by cycleForever() and wait until the thread has finished its tasks before exiting the application
    dust.cycle_stop()

def draw_detection(draw, d, c):
    coco_classes = {
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
    }
    width, height = draw.im.size
    # the box is relative to the image size so we multiply with height and width to get pixels
    top = d[0] * height
    left = d[1] * width
    bottom = d[2] * height
    right = d[3] * width
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
    right = min(width, np.floor(right + 0.5).astype('int32'))
    label = coco_classes[c]
    label_size = draw.textsize(label)
    if top - label_size[1] >= 0:
        text_origin = tuple(np.array([left, top - label_size[1]]))
    else:
        text_origin = tuple(np.array([left, top + 1]))
    color = ImageColor.getrgb("red")
    thickness = 0
    draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness],
                   outline=color)

    draw.text(text_origin, label, fill=color)

def postprocessingMobinet(num_detections, img, detection_classes, detection_boxes):
    batch_size = num_detections.shape[0]
    draw = ImageDraw.Draw(img)
    for batch in range(0, batch_size):
        for detection in range(0, int(num_detections[batch])):
            c = detection_classes[batch][detection]
            d = detection_boxes[batch][detection]
            draw_detection(draw, d, c)

    plt.figure(figsize=(80, 40))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def display_objdetect_image(image, boxes, labels, scores, score_threshold=0.7):
    # Resize boxes
    classes = [line.rstrip('\n') for line in open('labels.txt')]
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio

    _, ax = plt.subplots(1, figsize=(12, 9))
    image = np.array(image)
    ax.imshow(image)
    data = []
    # Showing boxes with score > 0.7
    for box, label, score in zip(boxes, labels, scores):

        if score > score_threshold:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b',
                                     facecolor='none')
            ax.annotate(classes[int(label)] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='b',
                        fontsize=12)
            ax.add_patch(rect)

            x1 = float(box[0])
            y1 = float(box[1])
            x2 = float(box[2])
            y2 = float(box[3])

            # print("mid point")
            # print((x_mid, y_mid))
            data.append({str(classes[int(label)]): {"x": x1, "y": y1, "z": 100, "width": x2 - x1, "height": y2 - y1}})
            # mid point formula x: x1+ (x2-x1)/2
            # mid point formula y: y1 + (y2-y1)/2

    payload = str(json.dumps(data)).encode("ascii")
    send_dust(payload)
    #plt.show()


def on_message(clientname, userdata, message):
    time.sleep(1)
    print("message received")
    data = json.loads(message.payload.decode('utf-8'))
    global choice
    choice = data['choice']
    global result1, result2, result3, disp, ogimg
    if choice == 1:
        result1 = np.array(data['data1']).astype('float32')
        result2= np.array(data['data2']).astype('float32')
        result3= np.array(data['data3']).astype('float32')
        disp=True
    elif choice == 2:
        global result4
        result1 = np.array(data['data1']).astype('float32')
        result2 = np.array(data['data2']).astype('float32')
        result3 = np.array(data['data3']).astype('float32')
        result4 = np.array(data['data4']).astype('float32')

        disp = True

def on_connect(mqtt_client, obj, flags, rc):
    if rc==0:
        client.subscribe("inference_out", qos=0)
        print("connected")
    else:
        print("connection refused")


broker = "127.0.0.1"
client = paho.Client("postprocessor")
client.on_message=on_message
client.on_connect=on_connect
client.connect(broker)
client.loop_start()
while 1:
    if disp:
        if choice == 1:
            display_objdetect_image(Image.open("demo.jpg"), result1, result2, result3)
            disp=False
        elif choice == 2:
            postprocessingMobinet(result4, Image.open("demo.jpg"), result2, result1)
            disp=False


