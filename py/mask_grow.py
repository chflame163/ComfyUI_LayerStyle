from .imagefunc import *

class MaskGrow:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "mask": ("MASK", ),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "grow": ("INT", {"default": 4, "min": -999, "max": 999, "step": 1}),
                "blur": ("INT", {"default": 4, "min": 0, "max": 999, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_grow'
    CATEGORY = '😺dzNodes/LayerMask'
    OUTPUT_NODE = True

    def mask_grow(self, mask, invert_mask, grow, blur,):

        if invert_mask:
            mask = 1 - mask
        # _mask = mask2image(mask).convert('L')

        ret_mask = expand_mask(mask, grow, blur)  # 扩张，模糊

        return (ret_mask,)

NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskGrow": MaskGrow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskGrow": "LayerMask: MaskGrow"
}