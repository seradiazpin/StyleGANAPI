from pydantic import BaseSettings


class Settings(BaseSettings):
    stylegan_network: str = "network/network-snapshot-008964.pkl"
    vgg_network: str = "network/vgg16_zhang_perceptual.pkl"
