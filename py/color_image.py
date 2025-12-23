from PIL import Image
from .imagefunc import log, pil2tensor


class ColorImage:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "height": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "color": ("STRING", {"default": "#000000"},),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = 'color_image'
    CATEGORY = '😺dzNodes/LayerUtility'

    def color_image(self, width, height, color, ):

        ret_image = Image.new('RGB', (width, height), color=color)
        return (pil2tensor(ret_image), )

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ColorImage": ColorImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ColorImage": "LayerUtility: ColorImage"
}