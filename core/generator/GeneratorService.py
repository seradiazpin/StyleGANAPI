import uuid

import PIL
import numpy as np

from core.image_storage.IImageStorage import IImageStorage
from core.network_operations.INetworkWrapper import INetworkWrapper
from core.training import misc
from data.firebase.firebase import FireBase


class GeneratorService:
    def __init__(self, network: INetworkWrapper, imageStorage: IImageStorage):
        self.network = network
        self.imageStorage = imageStorage

    def GenerateNewImage(self, seed, latents):
        imageExist = self.imageStorage.ImageAlreadyExist(seed)
        if imageExist is not None:
            return imageExist
        else:
            image_array, image_latents = self.network.GenerateImage(seed, latents)
            self.imageStorage.StoreImage(image_array, {"type_description": "generated", "type": "0", "seed": seed,
                                                       "latent": dict(enumerate(image_latents.tolist()))})
            exist = self.imageStorage.ImageAlreadyExist(seed)
            return exist

    def ProjectImage(self, image):
        image = PIL.Image.open(image).convert('RGB')
        image = image.resize((1024, 1024), PIL.Image.ANTIALIAS)
        image_array = np.array(image).swapaxes(0, 2).swapaxes(1, 2)
        image_array = misc.adjust_dynamic_range(image_array, [0, 255], [-1, 1])
        results, latents = self.network.ProjectImage(image_array)
        id_projection = uuid.uuid4()
        self.imageStorage.StoreImage(misc.convert_to_pil_image(misc.create_image_grid(results), drange=[-1, 1]),
                                     {"type_description": "project_image", "src_id": str(id_projection),
                                      "latent": dict(enumerate(latents.tolist())), "seed": str(id_projection)},
                                     is_PIL_image=True)

        exist = self.imageStorage.ImageAlreadyExist(str(id_projection))
        return exist

    def MixImage(self, src_seed=None, dst_seed=None, style_tag=0):
        id_mix = uuid.uuid4()
        id_src = uuid.uuid4()
        id_dst = uuid.uuid4()
        mix_seed = "{0}-{1}-{2}".format(src_seed, dst_seed, style_tag)
        src_data = self.imageStorage.ImageAlreadyExist(src_seed)
        dst_data = self.imageStorage.ImageAlreadyExist(dst_seed)
        mix_data = self.imageStorage.ImageAlreadyExist(mix_seed)

        if src_data is None or dst_data is None or mix_data is None:
            src_images, src_latents, dst_images, dst_latents, mix_image, mix_latents = self.network.MixImages(src_seed,
                                                                                                              dst_seed,
                                                                                                              style_tag)

            self.imageStorage.StoreImage(src_images,
                                         {"type_description": "mix_src", "src_id": str(id_src), "type": "2",
                                          "latent": dict(enumerate(src_latents.tolist())), "seed": src_seed})

            self.imageStorage.StoreImage(dst_images,
                                         {"type_description": "mix_dst",
                                          "dst_id": str(id_dst), "type": "2",
                                          "latent": dict(enumerate(dst_latents.tolist())), "seed": dst_seed})

            self.imageStorage.StoreImage(mix_image,
                                         {"type_description": "mix_rst", "mix_id": str(id_mix), "type": "2",
                                          "latent": dict(enumerate(mix_latents.tolist())), "seed": mix_seed})

            FireBase().create(u'Mixed', {"src_id": str(id_src), "dst_id": str(id_dst), "mix_id": str(id_mix),
                                         "type_description": "mix_images", "type": "2"})
        result = {"src": self.imageStorage.ImageAlreadyExist(src_seed),
                  "dst": self.imageStorage.ImageAlreadyExist(dst_seed),
                  "mix": self.imageStorage.ImageAlreadyExist(mix_seed)}
        return result

    def MixProjection(self, src_seed = None, id_image = None, style_tag=0):
        latent_vector = self.imageStorage.LoadImageLatents(id_image)
        if latent_vector is None:
            return None
        id_mix = uuid.uuid4()
        id_src = uuid.uuid4()
        id_dst = uuid.uuid4()
        mix_seed = "{0}-{1}-{2}".format(src_seed, id_image, style_tag)
        src_data = self.imageStorage.ImageAlreadyExist(src_seed)
        dst_data = self.imageStorage.ImageAlreadyExist(id_image)
        mix_data = self.imageStorage.ImageAlreadyExist(mix_seed)

        if src_data is None or dst_data is None or mix_data is None:
            src_images, src_latents, dst_images, dst_latents, mix_image, mix_latents = self.network.MixImages(src_seed,
                                                                                                              None,
                                                                                                              style_tag,
                                                                                                              latent_vector)

            self.imageStorage.StoreImage(src_images,
                                         {"type_description": "mix_src", "src_id": str(id_src), "type": "2",
                                          "latent": dict(enumerate(src_latents.tolist())), "seed": src_seed})

            self.imageStorage.StoreImage(dst_images,
                                         {"type_description": "mix_dst",
                                          "dst_id": str(id_dst), "type": "2",
                                          "latent": dict(enumerate(dst_latents.tolist())), "seed": id_image})

            self.imageStorage.StoreImage(mix_image,
                                         {"type_description": "mix_rst", "mix_id": str(id_mix), "type": "2",
                                          "latent": dict(enumerate(mix_latents.tolist())), "seed": mix_seed})

            FireBase().create(u'Mixed', {"src_id": str(id_src), "dst_id": str(id_dst), "mix_id": str(id_mix),
                                         "type_description": "mix_images", "type": "2"})
        result = {"src": self.imageStorage.ImageAlreadyExist(src_seed),
                  "dst": self.imageStorage.ImageAlreadyExist(id_image),
                  "mix": self.imageStorage.ImageAlreadyExist(mix_seed)}
        return result


