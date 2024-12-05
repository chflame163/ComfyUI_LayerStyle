from .imagefunc import AnyType, Hex_to_HSV_255level, log

any = AnyType("*")

class ColorValuetoHSVValue:

    def __init__(self):
        self.NODE_NAME = 'HSV Value'

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
    RETURN_NAMES = ("H", "S", "V")
    FUNCTION = 'color_value_to_hsv_value'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def color_value_to_hsv_value(self, color_value,):
        H, S, V = 0, 0, 0
        if isinstance(color_value, str):
            H, S, V = Hex_to_HSV_255level(color_value)
        elif isinstance(color_value, tuple):
            H, S, V = Hex_to_HSV_255level(RGB_to_Hex(color_value))
        else:
            log(f"{self.NODE_NAME}: color_value input type must be tuple or string.", message_type="error")

        return (H, S, V,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: HSV Value": ColorValuetoHSVValue
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: HSV Value": "LayerUtility: HSV Value"
}