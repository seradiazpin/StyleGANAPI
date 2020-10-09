from pydantic import BaseModel
from fastapi import UploadFile, File
from typing import Optional, List, Any


class GenerateParameters(BaseModel):
    seed: Optional[int]
    latent_vector: Optional[List[float]] = None


class MixParameters(BaseModel):
    seed1: Optional[int]
    seed2: Optional[int]
    style: Optional[int]

