from fastapi import APIRouter

from apis.gallery.gallery import gallery
from core.models.generate_parameters import GalleryPagination

router = APIRouter()


@router.post("/gallery/images", tags=["gallery"])
async def gallery_images(item: GalleryPagination):
    return gallery(item.page, item.size)["img"]
