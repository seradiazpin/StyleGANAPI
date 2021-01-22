from pydantic import BaseSettings


class Settings(BaseSettings):
    stylegan_network: str = "network/thuumnails.pkl"
    vgg_network: str = "network/vgg16_zhang_perceptual.pkl"
    firebase_config: str = "./data/firebase/firebasekey.json"
    firebase_bucket: str = "thumbnailgenerator-c1e1b.appspot.com"
    resolution: int = 1024
    original_w: int = 1920
    original_h: int = 1080
    small_w: int = 480
    small_h: int = 270



