import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor
from .imagefunc import mask_white_area, is_valid_mask



# 检查mask是否有效，如果mask面积少于指定比例则判为无效mask
class CheckMaskV2:

    def __init__(self):
        self.NODE_NAME = 'CheckMaskV2'
        pass

    @classmethod
    def INPUT_TYPES(self):
        method_list = ['simple', 'detect_percent']
        blank_mask_list = ['white', 'black']
        return {
            "required": {
                "mask": ("MASK",),  #
                "method": (method_list,),  #
                "white_point": ("INT", {"default": 1, "min": 1, "max": 254, "step": 1}), # 用于判断mask是否有效的白点值，高于此值被计入有效
                "area_percent": ("FLOAT", {"default": 0.01, "min": 0, "max": 100, "step": 0.01}), # 区域百分比，低于此则mask判定无效
            },
            "optional": { #
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ('bool',)
    FUNCTION = 'check_mask_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def check_mask_v2(self, mask, method, white_point, area_percent,):

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        tensor_mask = mask[0]

        pil_mask = tensor2pil(tensor_mask)
        if pil_mask.width * pil_mask.height > 262144:
            target_width = 512
            target_height = int(target_width * pil_mask.height / pil_mask.width)
            pil_mask = pil_mask.resize((target_width, target_height), Image.LANCZOS)
        ret_bool = False
        if method == 'simple':
            ret_bool = is_valid_mask(tensor_mask)
        else:
            ret_bool = mask_white_area(pil_mask, white_point) * 100 > area_percent
        log(f"{self.NODE_NAME}: {ret_bool}", message_type='finish')
        return (ret_bool,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: CheckMaskV2": CheckMaskV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: CheckMaskV2": "LayerUtility: Check Mask V2"
}