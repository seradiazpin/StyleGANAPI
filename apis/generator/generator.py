from typing import Dict
from core.generator.generator import generate


def generate_image(seed: int = 4444, latent: [] = []) -> str:
    img = generate()
    return img
