from abc import ABC


class INetworkWrapper(ABC):

    def GenerateImage(self):
        pass

    def ProjectImage(self):
        pass

    def MixImages(self):
        pass
