import PIL.Image

from config import Settings
from core.generator.GeneratorService import GeneratorService
from core.image_storage.FirebaseImageStorage import FirebaseImageStorage
from core.network_operations.StyleGANWrapper import StyleGANWrapper


def main():
    settings = Settings()
    latent = []
    generatorService = GeneratorService(StyleGANWrapper(settings.stylegan_network, settings.vgg_network)
                                        , FirebaseImageStorage(settings.original_w, settings.original_h
                                                               , settings.small_w, settings.small_h))
    image_url = generatorService.GenerateNewImage(4444, latent)
    print(image_url)
    image = PIL.Image.open('./static/example.jpg')
    generatorService = GeneratorService(StyleGANWrapper(settings.stylegan_network, settings.vgg_network)
                                        , FirebaseImageStorage(settings.original_w, settings.original_h
                                                               , settings.small_w, settings.small_h))
    img = generatorService.ProjectImage(image)
    print(img)


if __name__ == "__main__":
    main()
