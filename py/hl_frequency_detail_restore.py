import torch
from PIL import Image, ImageChops
from .imagefunc import log, tensor2pil, pil2tensor, chop_image_v2, gaussian_blur



class HLFrequencyDetailRestore:

    def __init__(self):
        self.NODE_NAME = 'HLFrequencyDetailRestore'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE",),
                "detail_image": ("IMAGE",),
                "keep_high_freq": ("INT", {"default": 64, "min": 0, "max": 1023}),
                "erase_low_freq": ("INT", {"default": 32, "min": 0, "max": 1023}),
                "mask_blur": ("INT", {"default": 16, "min": 0, "max": 1023}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'hl_frequency_detail_restore'
    CATEGORY = '😺dzNodes/LayerUtility'

    def hl_frequency_detail_restore(self, image, detail_image, keep_high_freq, erase_low_freq, mask_blur, mask=None):

        b_images = []
        l_images = []
        l_masks = []
        ret_images = []
        for b in image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in detail_image:
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
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        max_batch = max(len(b_images), len(l_images), len(l_masks))

        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            background_image = tensor2pil(background_image).convert('RGB')
            detail_image = l_images[i] if i < len(l_images) else l_images[-1]
            detail_image = tensor2pil(detail_image).convert('RGB')
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            high_ferq = chop_image_v2(ImageChops.invert(detail_image),
                                      gaussian_blur(detail_image, keep_high_freq),
                                      blend_mode='normal', opacity=50)
            high_ferq = ImageChops.invert(high_ferq)
            if erase_low_freq:
                low_freq = gaussian_blur(background_image, erase_low_freq)
            else:
                low_freq = background_image.copy()
            ret_image = chop_image_v2(low_freq, high_ferq, blend_mode="linear light", opacity=100)
            _mask = ImageChops.invert(_mask)
            if mask_blur > 0:
                _mask = gaussian_blur(_mask, mask_blur)
            ret_image.paste(background_image, _mask)
            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: HLFrequencyDetailRestore": HLFrequencyDetailRestore
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: HLFrequencyDetailRestore": "LayerUtility: H/L Frequency Detail Restore"
}