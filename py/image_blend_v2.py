import torch
import numpy as np
from PIL import Image
from .imagefunc import log, pil2tensor, tensor2pil, image2mask, mask2image, chop_image_v2, chop_mode_v2






class ImageBlendV2:

    def __init__(self):
        self.NODE_NAME = 'ImageBlendV2'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode_v2,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'image_blend_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_blend_v2(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity,
                  layer_mask=None
                  ):

        b_images = []
        l_images = []
        l_masks = []
        ret_images = []
        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in layer_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))
        if layer_mask is not None:
            if layer_mask.dim() == 2:
                layer_mask = torch.unsqueeze(layer_mask, 0)
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        max_batch = max(len(b_images), len(l_images), len(l_masks))
        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            layer_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            _canvas = tensor2pil(background_image).convert('RGB')
            _layer = tensor2pil(layer_image).convert('RGB')

            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                log(f"Warning: {self.NODE_NAME} mask mismatch, dropped!", message_type='warning')

            # 合成layer
            _comp = chop_image_v2(_canvas, _layer, blend_mode, opacity)
            _canvas.paste(_comp, mask=_mask)

            ret_images.append(pil2tensor(_canvas))
        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageBlend V2": ImageBlendV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageBlend V2": "LayerUtility: ImageBlend V2"
}