import json
import time
import uuid

import PIL

from core.util import encoder
from data.firebase.firebase import FireBase


def save_firebase(file, parameters={}):
    FireBase().create_file(u'Generated',
                      {"link": "generated/{0}.png".format(uuid.uuid4()), "tags": ["test"], "time": time.time(),
                       "path": file, "parameters": json.dumps(parameters)})


def save_image(images, parameters, file="./static/generated/example.png"):
    # Save image.
    new_img = PIL.Image.fromarray(images, 'RGB').resize(
        (1920, 1080), PIL.Image.ANTIALIAS)
    if file != "":
        PIL.Image.fromarray(images, 'RGB').resize((1920, 1080), PIL.Image.ANTIALIAS).save(file)
    save_firebase(file, parameters)
    return encoder.img_to_base64(new_img)


def save_PIL_image(image, parameters, file="./static/generated/example.png"):
    if file != "":
        image.resize((1920, 1080), PIL.Image.ANTIALIAS).save(file)
    return encoder.img_to_base64(image.resize((1920, 1080), PIL.Image.ANTIALIAS))
