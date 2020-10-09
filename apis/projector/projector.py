from typing import Dict
from core.projector.projector_generator import generate_projection, generate_projection_mix, mix_images


def project_image(image) -> str:
    img = generate_projection(image)
    return img


def chose_style_layers(styles: int) -> []:
    style = [range(8, 18)]
    if styles == 0:
        style = [range(0, 4)]
    if styles == 1:
        style = [range(4, 8)]
    if styles == 2:
        style = [range(8, 18)]
    return style


def mix(seed, seed2, styles) -> Dict[str, str]:
    style = chose_style_layers(styles)
    img = mix_images(seed, seed2, style)
    return img


def project_mix(seed, styles, image) -> Dict[str, str]:
    style = chose_style_layers(styles)
    img = generate_projection_mix(seed, style, image)
    return img
