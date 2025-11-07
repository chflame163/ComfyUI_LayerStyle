
from .imagefunc import Hex_to_RGB


class ColorPicker:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        mode_list = ['HEX', 'DEC']
        return {
            "required": {
                "color": ("COLOR", {"default": "#FFFFFF"},),
                "mode": (mode_list,),  # 输出模式
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = 'picker'
    CATEGORY = '😺dzNodes/LayerUtility'

    def picker(self, color, mode):
        ret = color
        if mode == 'DEC':
            ret = Hex_to_RGB(ret)
        return (ret,)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: ColorPicker": ColorPicker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ColorPicker": "LayerUtility: ColorPicker"
}