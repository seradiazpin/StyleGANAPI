from typing import Dict

from config import Settings
from core.generator.GeneratorService import GeneratorService
from core.image_storage.FirebaseImageStorage import FirebaseImageStorage
from core.network_operations.StyleGANWrapper import StyleGANWrapper

settings = Settings()


def project_image(image) -> Dict[str, str]:
    generatorService = GeneratorService(StyleGANWrapper(settings.stylegan_network, settings.vgg_network)
                                        , FirebaseImageStorage(settings.original_w, settings.original_h
                                                               , settings.small_w, settings.small_h))
    img = generatorService.ProjectImage(image)
    return img


def project_mix(seed, styles, id_image) -> Dict[str, str]:
    generatorService = GeneratorService(StyleGANWrapper(settings.stylegan_network, settings.vgg_network)
                                        , FirebaseImageStorage(settings.original_w, settings.original_h
                                                               , settings.small_w, settings.small_h))
    img = generatorService.MixProjection(seed, id_image)
    return img
