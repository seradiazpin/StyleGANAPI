from core.image_storage.IImageStorage import IImageStorage

import json
import time
import uuid

from PIL import Image
from data.firebase.firebase import FireBase


class FirebaseImageStorage(IImageStorage):
    def __init__(self, original_w, original_h, small_w, small_h, local_path="./static/generated/",
                 name_format="{0}-{1}.png"):
        self.name_format = name_format
        self.path = local_path
        self.original_w = original_w
        self.original_h = original_h
        self.small_w = small_w
        self.small_h = small_h
        self.firebase = FireBase()

    def GenerateFileNames(self):
        file_mame = {
            "original": self.name_format.format("original", "1920x1080"),
            "scaled": self.name_format.format("scaled", "480x270")
        }
        return self.path + "{0}", file_mame

    def StoreImage(self, image_data, parameters, path="./static/generated/", is_PIL_image=False):
        # Save image.
        file_path, file_mame = self.GenerateFileNames()
        if not is_PIL_image:
            Image.fromarray(image_data, 'RGB').resize((self.original_w, self.original_h), Image.ANTIALIAS).save(
                file_path.format(file_mame["original"]))
            Image.fromarray(image_data, 'RGB').resize((self.small_w, self.small_h), Image.ANTIALIAS).save(
                file_path.format(file_mame["scaled"]))
        else:
            image_data.resize((self.original_w, self.original_h), Image.ANTIALIAS).save(
                file_path.format(file_mame["original"]))
            image_data.resize((self.small_w, self.small_h), Image.ANTIALIAS).save(
                file_path.format(file_mame["scaled"]))
        image_id = uuid.uuid4()
        self.firebase.create_file(u'Generated',
                                  {"link": "generated/{0}X{1}".format(image_id, file_mame["original"]),
                                   "link_small": "generated/{0}X{1}".format(image_id, file_mame["scaled"]),
                                   "tags": ["test"],
                                   "time": time.time(),
                                   "path": file_path.format(file_mame["original"]),
                                   "path_small": file_path.format(file_mame["scaled"]),
                                   "seed": parameters["seed"],
                                   "parameters": json.dumps(parameters)})

    def ImageAlreadyExist(self, seed):
        fb = FireBase()
        data = fb.read_query(u'Generated', u'seed', u'==', seed)
        url = {}
        if len(data) != 0:
            for doc in data:
                url = doc.to_dict()
                url["link_small"] = fb.get_file_url(file=url["link_small"])
                url["link"] = fb.get_file_url(file=url["link"])
                url["id"] = doc.id
            return url
        return None

    def LoadImageLatents(self, image_id):
        data = self.firebase.read_query(u'Generated', u'seed', u'==', image_id)
        if len(data) != 0:
            response = json.loads(data[0].to_dict()["parameters"])["latent"]['0']
            return response
        return None
