from .imagefunc import *

NODE_NAME = ''

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

    def mask_grow(self, mask, invert_mask, grow, blur,):

        l_masks = []
        ret_masks = []

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)

        for m in mask:
            if invert_mask:
                m = 1 - m
            l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for i in range(len(l_masks)):

            _mask = l_masks[i]
            ret_masks.append(expand_mask(image2mask(_mask), grow, blur) )

        log(f"{NODE_NAME} Processed {len(ret_masks)} mask(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskGrow": MaskGrow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskGrow": "LayerMask: MaskGrow"
}