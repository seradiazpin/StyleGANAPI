import json
import time
import uuid

from PIL import Image

from core.util import encoder
from data.firebase.firebase import FireBase
from config import Settings
settings = Settings()

def file_name(path="./static/generated/"):
    file_mame = {
        "original": "{0}_{1}.png".format("original", "1920x1080"),
        "scaled": "{0}_{1}.png".format("scaled", "480x270")
    }
    return path + "{0}", file_mame


def save_firebase(file, file_names, parameters={}):
    image_id = uuid.uuid4()
    FireBase().create_file(u'Generated',
                           {"link": "generated/{0}X{1}".format(image_id, file_names["original"]),
                            "link_small": "generated/{0}X{1}".format(image_id, file_names["scaled"]),
                            "tags": ["test"],
                            "time": time.time(),
                            "path": file.format(file_names["original"]),
                            "path_small": file.format(file_names["scaled"]),
                            "seed":parameters["seed"],
                            "parameters": json.dumps(parameters)})
    return image_id


def save_image(images, parameters, path="./static/generated/"):
    # Save image.
    new_img = Image.fromarray(images, 'RGB').resize((settings.original_w, settings.original_h), Image.ANTIALIAS)
    file_path, file_mame = file_name(path)
    Image.fromarray(images, 'RGB').resize((settings.original_w, settings.original_h), Image.ANTIALIAS).save(file_path.format(file_mame["original"]))
    Image.fromarray(images, 'RGB').resize((settings.small_w, settings.small_h), Image.ANTIALIAS).save(file_path.format(file_mame["scaled"]))
    save_firebase(file_path, file_mame, parameters)


def save_PIL_image(image, parameters):
    file_path, file_mame = file_name()
    image.resize((settings.original_w, settings.original_h), Image.ANTIALIAS).save(file_path.format(file_mame["original"]))
    image.resize((settings.small_w, settings.small_h), Image.ANTIALIAS).save(file_path.format(file_mame["scaled"]))
    save_firebase(file_path, file_mame, parameters)
