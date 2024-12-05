import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, image2mask, create_mask_from_color_tensor, mask_fix




class MaskByColor:

    def __init__(self):
        self.NODE_NAME = 'MaskByColor'

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "color": ("COLOR", {"default": "#FFFFFF"},),
                "color_in_HEX": ("STRING", {"default": ""}),
                "threshold": ("INT", { "default": 50, "min": 0, "max": 100, "step": 1, }),
                "fix_gap": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1}),
                "fix_threshold": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 0.99, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mask_by_color"
    CATEGORY = '😺dzNodes/LayerMask'

    def mask_by_color(self, image, color, color_in_HEX, threshold,
                      fix_gap, fix_threshold, invert_mask, mask=None):

        if color_in_HEX != "" and color_in_HEX.startswith('#') and len(color_in_HEX) == 7:
            color = color_in_HEX

        ret_masks = []
        l_images = []
        l_masks = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for i in range(len(l_images)):
            img = l_images[i] if i < len(l_images) else l_images[-1]
            img = tensor2pil(img)
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            mask = Image.new('L', _mask.size, 'black')
            mask.paste(create_mask_from_color_tensor(img, color, threshold), mask=_mask)
            mask = image2mask(mask)
            if invert_mask:
                mask = 1 - mask
            if fix_gap:
                mask = mask_fix(mask, 1, fix_gap, fix_threshold, fix_threshold)
            ret_masks.append(mask)

        return (torch.cat(ret_masks, dim=0), )


NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskByColor": MaskByColor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskByColor": "LayerMask: Mask by Color"
}