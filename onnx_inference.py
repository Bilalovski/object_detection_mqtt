import json
import time

import onnxruntime
import paho.mqtt.client as paho
from matplotlib import pyplot as plt, patches
import numpy as np


def display_objdetect_image(image, boxes, labels, scores, score_threshold=0.7):
    # Resize boxes
    classes = [line.rstrip('\n') for line in open('labels.txt')]
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio

    _, ax = plt.subplots(1, figsize=(12, 9))
    image = np.array(image)
    ax.imshow(image)

    # Showing boxes with score > 0.7
    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b',
                                     facecolor='none')
            ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='b',
                        fontsize=12)
            ax.add_patch(rect)
    plt.show()


def on_message(clientName, userdata, message):
    time.sleep(1)
    print("message received")
    data = json.loads(message.payload.decode('utf-8'))
    choice = data['choice']
    if choice ==1:
        session = onnxruntime.InferenceSession("FasterRCNN-10.onnx")
        img_data = np.array(data['data']).astype('float32')
        print(img_data.shape)
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: np.array(img_data)})
        data1 = result[0].tolist()
        data2 = result[1].tolist()
        data3 = result[2].tolist()
        data = {'choice':choice, 'data1': data1, 'data2': data2, 'data3': data3}
        payload = json.dumps(data)
        client.publish("inference_out", payload)
        print("published data")
    elif choice==2:
        session = onnxruntime.InferenceSession("ssd_mobilenet_v1_10.onnx")
        img_data = np.array(data['data']).astype('uint8')
        print(img_data.shape)
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: np.array(img_data)})
        data1 = result[0].tolist()
        data2 = result[1].tolist()
        data3 = result[2].tolist()
        data4 = result[3].tolist()
        data = {'choice': choice, 'data1': data1, 'data2': data2, 'data3': data3, 'data4': data4}
        payload = json.dumps(data)
        client.publish("inference_out", payload)
        print("published data")
    # display_objdetect_image(Image.open("demo.jpg"), result[0], result[1], result[2])


def on_connect(mqtt_client, obj, flags, rc):
    if rc==0:
        client.subscribe("preprocess_out", qos=0)
        print("connected")
    else:
        print("connection refused")



broker = "127.0.0.1"
client = paho.Client("inference_node")
client.on_connect = on_connect
client.on_message = on_message
client.connect(broker)
client.loop_start()

while 1:
    pass
