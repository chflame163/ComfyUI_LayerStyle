import torch
from .imagefunc import log, AnyType, gradient, pil2tensor, tensor2pil, load_custom_size




any = AnyType("*")


class GradientImageV2:

    def __init__(self):
        self.NODE_NAME = 'GradientImage V2'

    @classmethod
    def INPUT_TYPES(self):
        size_list = ['custom']
        size_list.extend(load_custom_size())
        return {
            "required": {
                "size": (size_list,),
                "custom_width": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "custom_height": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "angle": ("INT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "start_color": ("STRING", {"default": "#FFFFFF"},),
                "end_color": ("STRING", {"default": "#000000"},),
            },
            "optional": {
                "size_as": (any, {}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = 'gradient_image_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def gradient_image_v2(self, size, custom_width, custom_height, angle, start_color, end_color, size_as=None):

        if size_as is not None:
            if size_as.shape[0] > 0:
                _asimage = tensor2pil(size_as[0])
            else:
                _asimage = tensor2pil(size_as)
            width, height = _asimage.size
        else:
            if size == 'custom':
                width = custom_width
                height = custom_height
            else:
                try:
                    _s = size.split('x')
                    width = int(_s[0].strip())
                    height = int(_s[1].strip())
                except Exception as e:
                    log(f'Warning: {self.NODE_NAME} invalid size, check "custom_size.ini"', message_type='warning')
                    width = custom_width
                    height = custom_height


        ret_image = gradient(start_color, end_color, width, height, angle)

        return (pil2tensor(ret_image), )

NODE_CLASS_MAPPINGS = {
    "LayerUtility: GradientImage V2": GradientImageV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: GradientImage V2": "LayerUtility: GradientImage V2"
}