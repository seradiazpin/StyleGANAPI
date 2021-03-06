from typing import Dict
from core.generator.GeneratorService import GeneratorService
from core.image_storage.FirebaseImageStorage import FirebaseImageStorage
from core.network_operations.StyleGANWrapper import StyleGANWrapper
from config import Settings

settings = Settings()


def generate_image(seed: int = 4444, latent=None) -> str:
    if latent is None:
        latent = []
    generatorService = GeneratorService(StyleGANWrapper(settings.stylegan_network, settings.vgg_network)
                                        , FirebaseImageStorage(settings.original_w, settings.original_h
                                                               , settings.small_w, settings.small_h))
    image_url = generatorService.GenerateNewImage(seed, latent)
    return image_url
