from .imagefunc import *

NODE_NAME = 'RGB Value'
any = AnyType("*")

class ColorValuetoRGBValue:

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

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("R", "G", "B")
    FUNCTION = 'color_value_to_rgb_value'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def color_value_to_rgb_value(self, color_value,):
        R, G, B = 0, 0, 0
        if isinstance(color_value, str):
            color = Hex_to_RGB(color_value)
            R, G, B = color[0], color[1], color[2]
        elif isinstance(color_value, tuple):
            R, G, B = color_value[0], color_value[1], color_value[2]
        else:
            log(f"{NODE_NAME}: color_value input type must be tuple or string.", message_type="error")

        return (R, G, B,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: RGB Value": ColorValuetoRGBValue
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: RGB Value": "LayerUtility: RGB Value"
}