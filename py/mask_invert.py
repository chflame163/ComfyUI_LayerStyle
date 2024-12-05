import torch
from PIL import Image
from .imagefunc import log, tensor2pil, image2mask, mask_invert



class MaskInvert:

    def __init__(self):
        self.NODE_NAME = 'MaskInvert'

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

    def mask_invert(self,mask):
        l_masks = []
        ret_masks = []

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)

        for m in mask:
            l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for i in range(len(l_masks)):
            _mask = l_masks[i]
            ret_masks.append(mask_invert(image2mask(_mask)))

        return (torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskInvert": MaskInvert
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskInvert": "LayerMask: MaskInvert"
}