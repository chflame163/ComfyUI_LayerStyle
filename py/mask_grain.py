import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, image2mask, expand_mask, chop_image_v2



class MaskGrain:

    def __init__(self):
        self.NODE_NAME = 'MaskGrain'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "mask": ("MASK", ),  #
                "grain": ("INT", {"default": 6, "min": 0, "max": 127, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_grain'
    CATEGORY = '😺dzNodes/LayerMask'

    def mask_grain(self, mask, grain, invert_mask):

        l_masks = []
        ret_masks = []

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        for m in mask:
            if invert_mask:
                m = 1 - m
            l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for mask in l_masks:
            if grain:
                white_mask = Image.new('L', mask.size, color="white")
                inner_mask = tensor2pil(expand_mask(image2mask(mask), 0 - grain, int(grain))).convert('L')
                outter_mask = tensor2pil(expand_mask(image2mask(mask), grain, int(grain * 2))).convert('L')
                ret_mask = Image.new('L', mask.size, color="black")
                ret_mask = chop_image_v2(ret_mask, outter_mask, blend_mode="dissolve", opacity=50).convert('L')
                ret_mask.paste(white_mask, mask=inner_mask)
                ret_masks.append(image2mask(ret_mask))
            else:
                ret_masks.append(image2mask(mask))

        log(f"{self.NODE_NAME} Processed {len(ret_masks)} mask(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskGrain": MaskGrain
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskGrain": "LayerMask: Mask Grain"
}