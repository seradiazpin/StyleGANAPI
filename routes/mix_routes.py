from fastapi import APIRouter

from apis.mix.mix import mix
from core.models.generate_parameters import MixParameters
from typing import Dict

router = APIRouter()
@router.post("/projector/mixSeeds", tags=["mix"])
async def mix_files(item: MixParameters) -> Dict[str, str]:
    return mix([item.seed1], [item.seed2], item.style)