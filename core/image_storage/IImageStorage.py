from abc import ABC


class IImageStorage(ABC):

    def StoreImage(self, image_array, parameters):
        pass

    def ImageAlreadyExist(self, seed):
        pass

    def LoadImageLatents(self, image_id):
        pass