import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor
from .imagefunc import mask_white_area




# 检查mask是否有效，如果mask面积少于指定比例则判为无效mask
class CheckMask:

    def __init__(self):
        self.NODE_NAME = 'CheckMask'


    @classmethod
    def INPUT_TYPES(self):
        blank_mask_list = ['white', 'black']
        return {
            "required": {
                "mask": ("MASK",),  #
                "white_point": ("INT", {"default": 1, "min": 1, "max": 254, "step": 1}), # 用于判断mask是否有效的白点值，高于此值被计入有效
                "area_percent": ("INT", {"default": 1, "min": 1, "max": 99, "step": 1}), # 区域百分比，低于此则mask判定无效
            },
            "optional": { #
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ('bool',)
    FUNCTION = 'check_mask'
    CATEGORY = '😺dzNodes/LayerUtility'

    def check_mask(self, mask, white_point, area_percent,):

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        mask = tensor2pil(mask[0])
        if mask.width * mask.height > 262144:
            target_width = 512
            target_height = int(target_width * mask.height / mask.width)
            mask = mask.resize((target_width, target_height), Image.LANCZOS)
        ret = mask_white_area(mask, white_point) * 100 > area_percent
        log(f"{self.NODE_NAME}:{ret}", message_type="finish")
        return (ret,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: CheckMask": CheckMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: CheckMask": "LayerUtility: Check Mask"
}