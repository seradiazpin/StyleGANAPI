from config import Settings
from core.generator.GeneratorService import GeneratorService
from typing import Dict

from core.image_storage.FirebaseImageStorage import FirebaseImageStorage
from core.network_operations.StyleGANWrapper import StyleGANWrapper

settings = Settings()


def mix(seed, seed2, styles) -> Dict[str, str]:
    generatorService = GeneratorService(StyleGANWrapper(settings.stylegan_network, settings.vgg_network)
                                        , FirebaseImageStorage(settings.original_w, settings.original_h
                                                               , settings.small_w, settings.small_h))
    image_urls = generatorService.MixImage(seed, seed2, styles)
    return image_urls
