from .imagefunc import *

class MaskStrkoe:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "mask": ("MASK", ),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "stroke_grow": ("INT", {"default": 0, "min": -999, "max": 999, "step": 1}),  # 收缩值
                "stroke_width": ("INT", {"default": 20, "min": 0, "max": 999, "step": 1}),  # 扩张值
                "blur": ("INT", {"default": 6, "min": 0, "max": 100, "step": 1}),  # 模糊
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_stroke'
    CATEGORY = '😺dzNodes/LayerMask'
    OUTPUT_NODE = True

    def mask_stroke(self, mask, invert_mask, stroke_grow, stroke_width, blur,):

        l_masks = []
        ret_masks = []

        for m in mask:
            if invert_mask:
                m = 1 - m
            l_masks.append(tensor2pil(m).convert('L'))

        for i in range(len(l_masks)):
            _mask = l_masks[i]
            grow_offset = int(stroke_width / 2)
            inner_stroke = stroke_grow - grow_offset
            outer_stroke = inner_stroke + stroke_width
            inner_mask = expand_mask(image2mask(_mask), inner_stroke, blur)
            outer_mask = expand_mask(image2mask(_mask), outer_stroke, blur)
            stroke_mask = subtract_mask(outer_mask, inner_mask)
            ret_masks.append(stroke_mask)

        log(f'MaskStrkoe Processed {len(ret_masks)} image(s).')
        return (torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskStrkoe": MaskStrkoe
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskStrkoe": "LayerMask: MaskStrkoe"
}