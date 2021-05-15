from fastapi import APIRouter, UploadFile, File

from typing import Dict
from apis.projector.projector import project_image, project_mix
from core.models.generate_parameters import ProjectionMixParameters

router = APIRouter()


@router.post("/projector/image", tags=["projector"])
async def project_image_file(file: UploadFile = File(...)) -> Dict[str, str]:
    return project_image(file.file)


@router.post("/projector/mixProjection", tags=["projector"])
async def mix_projection(item: ProjectionMixParameters) -> Dict[str, str]:
    return project_mix([item.seed1], item.style, item.id_image)
