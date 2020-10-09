from typing import Dict
from apis.generator.generator import generate_image
from apis.projector.projector import project_image, project_mix, mix
from fastapi import APIRouter, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

from core.models.generate_parameters import MixParameters, GenerateParameters

router = APIRouter()


@router.get("/generator/image", tags=["generator"])
async def generate_image_bs64() -> str:
    return generate_image()


@router.post("/generator/imageByVector", tags=["generator"])
def generate_image_latent_vector(item: GenerateParameters)-> str:
    return generate_image(item.seed, item.latent_vector)


@router.post("/projector/image", tags=["projector"])
async def project_image_bs64(file: UploadFile = File(...)) -> str:
    return project_image(file.file)


@router.post("/projector/mixSeeds", tags=["projector"])
async def mix_seeds_bs64(item: MixParameters) -> Dict[str, str]:
    return mix([item.seed1], [item.seed2], item.style)


@router.post("/projector/mixProjection", tags=["projector"])
async def mix_projection_bs64(seed1: int = Form(...), seed2: int = Form(...), style: int = Form(...),
                              file: UploadFile = File(...)) -> Dict[str, str]:
    return project_mix([seed1], style, file.file)


@router.get("/files/generate/", tags=["debug"])
async def generate_image_file():
    generate_image()
    return FileResponse("static/generated/example.png")


@router.post("/files/mix/", tags=["debug"])
async def project_image_file(file: UploadFile = File(...)):
    project_mix([999], [range(8, 18)], file.file)
    return FileResponse("static/mix/dst.png")


@router.post("/files/project/", tags=["debug"])
async def project_mix_file(file: UploadFile = File(...)):
    project_image(file.file)
    return FileResponse("static/projected/project-last.png")
