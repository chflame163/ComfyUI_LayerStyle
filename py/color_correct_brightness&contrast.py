from PIL import ImageEnhance
from .imagefunc import *

NODE_NAME = 'Brightness & Contrast'

class ColorCorrectBrightnessAndContrast:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "brightness": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_brightness_and_contrast'
    CATEGORY = '😺dzNodes/LayerColor'
    OUTPUT_NODE = True

    def color_correct_brightness_and_contrast(self, image, brightness, contrast, saturation):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i,0)

            _image = tensor2pil(i).convert('RGB')
            if brightness != 1:
                brightness_image = ImageEnhance.Brightness(_image)
                _image = brightness_image.enhance(factor=brightness)
            if contrast != 1:
                contrast_image = ImageEnhance.Contrast(_image)
                _image = contrast_image.enhance(factor=contrast)
            if saturation != 1:
                color_image = ImageEnhance.Color(_image)
                _image = color_image.enhance(factor=saturation)
            ret_images.append(pil2tensor(_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).")
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerColor: Brightness & Contrast": ColorCorrectBrightnessAndContrast
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: Brightness & Contrast": "LayerColor: Brightness & Contrast"
}