import time

from pydust import core

from PIL import Image
import pickle
import json
def receive(arg):
    print(json.loads(arg.decode('ascii')))


def main():
    # initialises the core with the given block name and the directory where the modules are located (default "./modules")
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
    dust.register_listener("check_output", receive)
    #dust.register_listener("subscribe-mqtt", lambda payload: print("Received payload with %d bytes" % len(payload)))

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()