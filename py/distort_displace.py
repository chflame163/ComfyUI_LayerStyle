import copy
import torch
import numpy as np
from .imagefunc import log, pil2tensor, tensor2pil, chop_image_v2, chop_mode_v2, fit_resize_image, displacement_image


class LS_DistortDisplace:

    def __init__(self):
        self.NODE_NAME = 'DistortDisplace'

    @classmethod
    def INPUT_TYPES(self):
        shadow_blendmode_list = ['linear burn', "multiply"]
        highlight_blendmode_list = ['screen', 'linear dodge(add)']
        shadow_blendmode_list = shadow_blendmode_list + [x for x in chop_mode_v2 if x not in shadow_blendmode_list]
        highlight_blendmode_list = highlight_blendmode_list + [x for x in chop_mode_v2 if x not in highlight_blendmode_list]
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "material_image": ("IMAGE",),  #
                "distort_strength": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.1}),
                "smoothness": ("INT", {"default": 8, "min": 0, "max": 99, "step": 1}),
                "anti_aliasing": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "shadow_blend_mode": (shadow_blendmode_list,),
                "shadow_strength": ("INT", {"default": 75, "min": 0, "max": 100, "step": 1}),  # 透明度
                "highlight_blend_mode": (highlight_blendmode_list,),
                "highlight_strength": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("image", "displaced_material")
    FUNCTION = 'distort_displace'
    CATEGORY = '😺dzNodes/LayerFilter'

    def distort_displace(self, image, material_image, distort_strength, smoothness, anti_aliasing,
                              shadow_blend_mode, shadow_strength, highlight_blend_mode, highlight_strength,
                              mask=None):

        m_images = []
        i_images = []
        i_masks = []
        ret_images = []
        displaced_images = []

        for m in material_image:
            m_images.append(torch.unsqueeze(m, 0))
        for i in image:
            i_images.append(torch.unsqueeze(i, 0))
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            for m in mask:
                i_masks.append(torch.unsqueeze(m, 0))

        max_batch = max(len(m_images), len(i_images), len(i_masks))
        for i in range(max_batch):
            m_img = m_images[i] if i < len(m_images) else m_images[-1]
            _image = tensor2pil(m_img).convert('RGB')
            i_img = i_images[i] if i < len(i_images) else i_images[-1]
            _grayscale = tensor2pil(i_img)

            log(f"{self.NODE_NAME} processing:")

            displaced_image = displacement_image(_image, _grayscale, distort_strength, smoothness, anti_aliasing)

            orig_image = tensor2pil(i_img)
            if shadow_strength > 0:
                ret_image = chop_image_v2(orig_image, displaced_image, shadow_blend_mode, shadow_strength)
            else:
                ret_image = orig_image
            if highlight_strength > 0:
                ret_image = chop_image_v2(ret_image, displaced_image, highlight_blend_mode, highlight_strength)

            if mask is not None:
                i_msk = i_masks[i] if i < len(i_masks) else i_masks[-1]
                _mask = tensor2pil(i_msk).convert('L')
                if _mask.size != displaced_image.size:
                    _mask = fit_resize_image(_mask, displaced_image.width, displaced_image.height,'fill', Image.LANCZOS)
                    log(f"Warning: {self.NODE_NAME} mask mismatch, fixed to image size!", message_type='warning')
                orig_image.paste(ret_image, mask=_mask)
                ret_image = orig_image

            ret_images.append(pil2tensor(ret_image))
            displaced_images.append(pil2tensor(displaced_image))
            log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')

        return (torch.cat(ret_images, dim=0), torch.cat(displaced_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerFilter: DistortDisplace": LS_DistortDisplace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: DistortDisplace": "LayerFilter: Distort Displace",
}
