import torch
from .imagefunc import log, tensor2pil, pil2tensor
from .imagefunc import image_hue_offset, image_gray_offset, image_channel_merge, RGB2RGBA



class ColorCorrectHSV:

    def __init__(self):
        self.NODE_NAME = 'HSV'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "H": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "S": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "V": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_HSV'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_HSV(self, image, H, S, V):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i,0)
            __image = tensor2pil(i)
            _h, _s, _v = tensor2pil(i).convert('HSV').split()
            if H != 0 :
                _h = image_hue_offset(_h, H)
            if S != 0 :
                _s = image_gray_offset(_s, S)
            if V != 0 :
                _v = image_gray_offset(_v, V)
            ret_image = image_channel_merge((_h, _s, _v), 'HSV')

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerColor: HSV": ColorCorrectHSV
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: HSV": "LayerColor: HSV"
}