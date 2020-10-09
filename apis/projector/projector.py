from typing import Dict
from core.projector.projector_generator import generate_projection, generate_projection_mix


def project_image(image) -> Dict[str, str]:
    img = generate_projection(image)
    return {"img": img}


def project_mix(seed, styles, image) -> Dict[str, str]:
    img = generate_projection_mix(seed, styles, image)
    return {"img": img}
