import enum

class MaskType(enum.Enum):
    PositiveWhite = "positive_white"
    PositiveBlack = "positive_black"

class MaskImage:
    def __init__(self, mask, mask_type: MaskType):
        self.mask = mask
        self.mask_type = mask_type

    def get_positive_black_image(self):
        if self.mask_type == MaskType.PositiveWhite:
            raise Exception("aa")
        elif self.mask_type == MaskType.PositiveBlack:
            return self.mask