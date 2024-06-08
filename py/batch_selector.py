
from .imagefunc import *

NODE_NAME = 'BatchSelector'



class BatchSelector:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "select": ("STRING", {"default": "0,"},),
            },
            "optional": {
                "images": ("IMAGE",),  #
                "masks": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = 'batch_selector'
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'

    def batch_selector(self, select, images=None, masks=None
                  ):
        ret_images = []
        ret_masks = []
        empty_image = pil2tensor(Image.new("RGBA", (64, 64), (0, 0, 0, 0)))
        empty_mask = image2mask(Image.new("L", (64, 64), color="black"))

        indexs = extract_numbers(select)
        for i in indexs:
            if images is not None:
                if i < len(images):
                    ret_images.append(images[i].unsqueeze(0))
                else:
                    ret_images.append(images[-1].unsqueeze(0))
            if masks is not None:
                if i < len(masks):
                    ret_masks.append(masks[i].unsqueeze(0))
                else:
                    ret_masks.append(masks[-1].unsqueeze(0))

        if len(ret_images) == 0:
            ret_images.append(empty_image)
        if len(ret_masks) == 0:
            ret_masks.append(empty_mask)

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: BatchSelector": BatchSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: BatchSelector": "LayerUtility: Batch Selector"
}