from apis.generator.generator import generate_image
from apis.mix.mix import mix
from apis.projector.projector import project_image
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from typing import Dict
from core.models.generate_parameters import MixParameters

router = APIRouter()


@router.get("/files/generate/", tags=["debug"])
async def generate_image_file():
    img = generate_image()
    return img


@router.post("/files/mix/", tags=["debug"])
async def project_image_file(item: MixParameters) -> Dict[str, str]:
    img = mix([item.seed1], [item.seed2], item.style)
    return img


@router.post("/files/project/", tags=["debug"])
async def project_mix_file(file: UploadFile = File(...)):
    img = project_image(file.file)
    return img


@router.get("/", tags=["webpage"])
async def index():
    return FileResponse('webpage/index.html')
