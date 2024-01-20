from .imagefunc import *

class MaskInvert:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "mask": ("MASK", ),  #
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_invert'
    CATEGORY = '😺dzNodes/LayerMask'
    OUTPUT_NODE = True

    def mask_invert(self,mask):

        return (mask_invert(mask),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskInvert": MaskInvert
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskInvert": "LayerMask: MaskInvert"
}