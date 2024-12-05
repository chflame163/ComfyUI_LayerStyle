import torch
import numpy as np
from PIL import Image, ImageEnhance
from .imagefunc import log, tensor2pil, pil2tensor
from .imagefunc import RGB2RGBA



class ColorCorrectExposure:

    def __init__(self):
        self.NODE_NAME = 'Exposure'
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "exposure": ("INT", {"default": 20, "min": -100, "max": 100, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_exposure'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_exposure(self, image, exposure):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            __image = tensor2pil(i)
            t = i.detach().clone().cpu().numpy().astype(np.float32)
            more = t[:, :, :, :3] > 0
            t[:, :, :, :3][more] *= pow(2, exposure / 32)
            if exposure < 0:
                bp = -exposure / 250
                scale = 1 / (1 - bp)
                t = np.clip((t - bp) * scale, 0.0, 1.0)
            ret_image = tensor2pil(torch.from_numpy(t))

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerColor: Exposure": ColorCorrectExposure
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: Exposure": "LayerColor: Exposure"
}