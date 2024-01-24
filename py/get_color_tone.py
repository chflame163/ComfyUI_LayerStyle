from .imagefunc import *

class GetColorTone:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        mode_list = ['main_color', 'average']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mode": (mode_list,),  # 主色/平均色
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RGB color in HEX",)
    FUNCTION = 'get_color_tone'
    CATEGORY = '😺dzNodes/LayerUtility'
    OUTPUT_NODE = True

    def get_color_tone(self, image, mode,):

        _canvas = tensor2pil(image).convert('RGB')
        _canvas = gaussian_blur(_canvas, int((_canvas.width + _canvas.height) / 200))
        if mode == 'main_color':
            ret_color = get_image_color_tone(_canvas)
        else:
            ret_color = get_image_color_average(_canvas)

        return (ret_color,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: GetColorTone": GetColorTone
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: GetColorTone": "LayerUtility: GetColorTone"
}