import torch
from .imagefunc import log, tensor2pil, pil2tensor
from .imagefunc import gamma_trans, RGB2RGBA



class ColorCorrectGamma:

    def __init__(self):
        self.NODE_NAME = 'Gamma'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
                "gamma": ("FLOAT", {"default": 1, "min": 0.1, "max": 10, "step": 0.01}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_gamma'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_gamma(self, image, gamma):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            __image = tensor2pil(i)
            ret_image = gamma_trans(tensor2pil(i), gamma)

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerColor: Gamma": ColorCorrectGamma
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: Gamma": "LayerColor: Gamma"
}