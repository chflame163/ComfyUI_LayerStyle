from .imagefunc import *

NODE_NAME = 'MaskInvert'

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
        l_masks = []
        ret_masks = []

        for m in mask:
            l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for i in range(len(l_masks)):
            _mask = l_masks[i]
            ret_masks.append(mask_invert(image2mask(_mask)))

        log(f"{NODE_NAME} Processed {len(ret_masks)} image(s).")
        return (torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskInvert": MaskInvert
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskInvert": "LayerMask: MaskInvert"
}