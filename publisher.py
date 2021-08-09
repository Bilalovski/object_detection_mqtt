import time
from PIL import Image
from pydust import core

import pickle


def main():
    # initialises the core with the given block name and the directory where the modules are located (default "./modules")
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
    img = Image.open("demo.jpg")
    img_bytes = pickle.dumps(img)

    # publishes the payload to the given channel (as defined by the configuration file)
    i = 0
    dust.publish("publish-mqtt", img_bytes)

    time.sleep(1)

    # disconnects all channels and flushes the addon stack and transport.
    dust.disconnect()

    # stops the background thread started by cycleForever() and wait until the thread has finished its tasks before exiting the application
    dust.cycle_stop()


if __name__ == "__main__":
    main()
