from pydantic import BaseModel
from typing import Optional, List


class GenerateParameters(BaseModel):
    seed: Optional[int]
    latent_vector: Optional[List[float]] = None


class MixParameters(BaseModel):
    seed1: Optional[int]
    seed2: Optional[int]
    style: Optional[int]
