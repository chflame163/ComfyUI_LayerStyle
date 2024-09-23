from .imagefunc import *

NODE_NAME = 'Gray Value'
any = AnyType("*")

class ColorValuetoGrayValue:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "color_value": (any, {}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("gray(256_level)", "gray(100_level)",)
    FUNCTION = 'color_value_to_gray_value'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def color_value_to_gray_value(self, color_value,):
        gray = rgb2gray(color_value)
        return (gray, int(gray / 2.55),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: GrayValue": ColorValuetoGrayValue
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: GrayValue": "LayerUtility: Gray Value"
}