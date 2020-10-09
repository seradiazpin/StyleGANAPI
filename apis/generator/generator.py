from typing import Dict
from core.generator.generator import generate


def main_func() -> Dict[str, str]:
    img = generate()
    return {"img": img}
