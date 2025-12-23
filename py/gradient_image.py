import torch
from .imagefunc import gradient, pil2tensor

class GradientImage:

    def __init__(self):
        self.NODE_NAME = 'GradientImage'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "height": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "angle": ("INT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "start_color": ("STRING", {"default": "#FFFFFF"},),
                "end_color": ("STRING", {"default": "#000000"},),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = 'gradient_image'
    CATEGORY = '😺dzNodes/LayerUtility'

    def gradient_image(self, width, height, angle, start_color, end_color, ):

        ret_image = gradient(start_color, end_color, width, height, angle)

        return (pil2tensor(ret_image), )

NODE_CLASS_MAPPINGS = {
    "LayerUtility: GradientImage": GradientImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: GradientImage": "LayerUtility: GradientImage"
}