from fastapi import APIRouter

from apis.generator.generator import generate_image
from core.models.generate_parameters import GenerateParameters

router = APIRouter()


@router.get("/generator/image", tags=["generator"])
async def generate_image_bs64() -> str:
    return generate_image()


@router.post("/generator/imageByVector", tags=["generator"])
def generate_image_latent_vector(item: GenerateParameters) -> str:
    return generate_image(item.seed, item.latent_vector)
