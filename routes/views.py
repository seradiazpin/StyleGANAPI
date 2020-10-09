from typing import Dict
from apis.generator.generator import main_func as main_func_a
from apis.projector.projector import project_image, project_mix
from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()


@router.get("/api_a/{num}", tags=["api_a"])
async def view_a(num: int) -> Dict[str, str]:
    return main_func_a()


@router.get("/api_b/{num}", tags=["api_b"])
async def view_b(num: int) -> Dict[str, str]:
    return project_image()


@router.get("/files/generate/", tags=["debug"])
def generate_image_file():
    main_func_a()
    return FileResponse("static/generated/example.png")


@router.post("/files/mix/", tags=["debug"])
def project_image_file(file: UploadFile = File(...)):
    project_mix([999], [range(8, 18)], file.file)
    return FileResponse("static/mix/dst.png")


@router.post("/files/project/", tags=["debug"])
def project_mix_file(file: UploadFile = File(...)):
    project_image(file.file)
    return FileResponse("static/projected/project-last.png")
