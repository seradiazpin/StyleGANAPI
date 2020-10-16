from typing import Dict
from core.gallery.gallery import load_gallery


def gallery(document, size):
    img = load_gallery(document, size)
    return img
