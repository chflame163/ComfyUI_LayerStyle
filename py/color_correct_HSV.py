from .imagefunc import *

class ColorCorrectHSV:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "H": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "S": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "V": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_HSV'
    CATEGORY = '😺dzNodes/LayerColor'
    OUTPUT_NODE = True

    def color_correct_HSV(self, image, H, S, V):

        _h, _s, _v = tensor2pil(image).convert('HSV').split()
        if H != 0 :
            _h = image_hue_offset(_h, H)
        if S != 0 :
            _s = image_gray_offset(_s, S)
        if V != 0 :
            _v = image_gray_offset(_v, V)
        ret_image = image_channel_merge((_h, _s, _v), 'HSV')

        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerColor: HSV": ColorCorrectHSV
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: HSV": "LayerColor: HSV"
}