import torch
from .imagefunc import log, tensor2pil, pil2tensor
from .imagefunc import image_gray_offset, image_channel_merge, RGB2RGBA



class ColorCorrectLAB:

    def __init__(self):
        self.NODE_NAME = 'LAB'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "L": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "A": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "B": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_LAB'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_LAB(self, image, L, A, B):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            __image = tensor2pil(i)
            _l, _a, _b = tensor2pil(i).convert('LAB').split()
            if L != 0 :
                _l = image_gray_offset(_l, L)
            if A != 0 :
                _a = image_gray_offset(_a, A)
            if B != 0 :
                _b = image_gray_offset(_b, B)
            ret_image = image_channel_merge((_l, _a, _b), 'LAB')

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerColor: LAB": ColorCorrectLAB
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: LAB": "LayerColor: LAB"
}