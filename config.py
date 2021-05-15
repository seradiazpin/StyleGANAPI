from pydantic import BaseSettings

class Settings(BaseSettings):
    stylegan_network: str = "network/thumbnails.pkl"
    vgg_network: str = "network/vgg16_zhang_perceptual.pkl"
    firebase_config: str = "./data/firebase/firebasekey.json"
    firebase_bucket: str = "thumbnailgenerator-c1e1b.appspot.com"
    resolution: int = 1024
    original_w: int = 1920
    original_h: int = 1080
    small_w: int = 480
    small_h: int = 270

"""
class Settings(BaseSettings):
    stylegan_network: str = "network/subaruport.pkl"
    vgg_network: str = "network/vgg16_zhang_perceptual.pkl"
    firebase_config: str = "./data/firebase/anime-faces.json"
    firebase_bucket: str = "anime-faces-dc87a.appspot.com"
    resolution: int = 512
    original_w: int = 512
    original_h: int = 512
    small_w: int = 128
    small_h: int = 128
"""