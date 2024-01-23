from .imagefunc import *

class GaussianBlur:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "blur": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),  # 模糊
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'gaussian_blur'
    CATEGORY = '😺dzNodes/LayerFilter'
    OUTPUT_NODE = True

    def gaussian_blur(self, image, blur):

        _canvas = tensor2pil(image).convert('RGB')
        ret_image = gaussian_blur(_canvas, blur)
        log('GaussianBlur Processed.')

        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerFilter: GaussianBlur": GaussianBlur
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: GaussianBlur": "LayerFilter: GaussianBlur"
}