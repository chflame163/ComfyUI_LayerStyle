import torch
from PIL import Image, ImageEnhance
from .imagefunc import log, tensor2pil, pil2tensor
from .imagefunc import color_balance, RGB2RGBA




class ColorBalance:

    def __init__(self):
        self.NODE_NAME = 'ColorBalance'


    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "cyan_red": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "magenta_green": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "yellow_blue": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_balance'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_balance(self, image, cyan_red, magenta_green, yellow_blue):

        l_images = []
        l_masks = []
        ret_images = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))


        for i in range(len(l_images)):
            _image = l_images[i]
            _mask = l_masks[i]
            orig_image = tensor2pil(_image)

            ret_image = color_balance(orig_image,
                              [cyan_red, magenta_green, yellow_blue],
                              [cyan_red, magenta_green, yellow_blue],
                              [cyan_red, magenta_green, yellow_blue],
                                      shadow_center=0.15,
                                      midtone_center=0.5,
                                      midtone_max=1,
                                      preserve_luminosity=True)

            if orig_image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, orig_image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerColor: ColorBalance": ColorBalance
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: ColorBalance": "LayerColor: ColorBalance"
}