import copy

from .imagefunc import *

class MaskMotionBlur:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blur": ("INT", {"default": 20, "min": 1, "max": 9999, "step": 1}),
                "angle": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_motion_blur'
    CATEGORY = '😺dzNodes/LayerMask'
    OUTPUT_NODE = True

    def mask_motion_blur(self, mask, invert_mask, blur, angle,):

        if invert_mask:
            mask = 1 - mask
        _mask = mask2image(mask).convert('RGB')
        _blurimage = motion_blur(_mask, angle, blur)
        ret_mask = image2mask(_blurimage)
        return (ret_mask,)

NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskMotionBlur": MaskMotionBlur
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskMotionBlur": "LayerMask: MaskMotionBlur"
}