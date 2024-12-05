import torch
from .imagefunc import log, tensor2pil, pil2tensor
from .imagefunc import image_gray_offset, image_channel_merge, RGB2RGBA



class ColorCorrectYUV:

    def __init__(self):
        self.NODE_NAME = 'YUV'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "Y": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "U": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "V": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_YUV'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_YUV(self, image, Y, U, V):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            __image = tensor2pil(i)
            _y, _u, _v = tensor2pil(i).convert('YCbCr').split()
            if Y != 0 :
                _y = image_gray_offset(_y, Y)
            if U != 0 :
                _u = image_gray_offset(_u, U)
            if V != 0 :
                _v = image_gray_offset(_v, V)
            ret_image = image_channel_merge((_y, _u, _v), 'YCbCr')

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerColor: YUV": ColorCorrectYUV
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: YUV": "LayerColor: YUV"
}