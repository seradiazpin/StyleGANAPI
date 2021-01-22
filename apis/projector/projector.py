from typing import Dict
from core.projector.projector_generator import generate_projection
from core.mixer.mixer import generate_projection_mix, mix_images, generate_projection_mix_old


def project_image(image) -> Dict[str, str]:
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
    img = mix_images(seed, seed2, style, styles)
    return img


def project_mix(seed, styles, id_image) -> Dict[str, str]:
    style = chose_style_layers(styles)
    img = generate_projection_mix(seed, style, id_image, styles)
    return img


def project_mix_old(seed, styles, image) -> Dict[str, str]:
    style = chose_style_layers(styles)
    img = generate_projection_mix_old(seed, style, image,styles )
    return img
