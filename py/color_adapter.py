from .imagefunc import *

class ColorAdapter:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "color_ref_image": ("IMAGE", ),  #
                "opacity": ("INT", {"default": 75, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_adapter'
    CATEGORY = '😺dzNodes/LayerColor'
    OUTPUT_NODE = True

    def color_adapter(self, image, color_ref_image, opacity):
        _canvas = tensor2pil(image).convert('RGB')
        ret_image = color_adapter(_canvas, tensor2pil(color_ref_image).convert('RGB'))
        ret_image = chop_image(_canvas, ret_image, blend_mode='normal', opacity=opacity)
        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerColor: ColorAdapter": ColorAdapter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: ColorAdapter": "LayerColor: ColorAdapter"
}