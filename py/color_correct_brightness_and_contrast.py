from .imagefunc import *



class ColorCorrectBrightnessAndContrast:

    def __init__(self):
        self.NODE_NAME = 'Brightness & Contrast'

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

    def color_correct_brightness_and_contrast(self, image, brightness, contrast, saturation):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i,0)
            __image = tensor2pil(i)
            ret_image = __image.convert('RGB')
            if brightness != 1:
                brightness_image = ImageEnhance.Brightness(ret_image)
                ret_image = brightness_image.enhance(factor=brightness)
            if contrast != 1:
                contrast_image = ImageEnhance.Contrast(ret_image)
                ret_image = contrast_image.enhance(factor=contrast)
            if saturation != 1:
                color_image = ImageEnhance.Color(ret_image)
                ret_image = color_image.enhance(factor=saturation)

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])
            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)


# 节点名称去掉“&”
class LS_ColorCorrect_Brightness_And_Contrast_V2:
    def __init__(self):
        self.NODE_NAME = 'Brightness Contrast V2'

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
    FUNCTION = 'color_correct_brightness_contrast_v2'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_brightness_contrast_v2(self, image, brightness, contrast, saturation):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i,0)
            __image = tensor2pil(i)
            ret_image = __image.convert('RGB')
            if brightness != 1:
                brightness_image = ImageEnhance.Brightness(ret_image)
                ret_image = brightness_image.enhance(factor=brightness)
            if contrast != 1:
                contrast_image = ImageEnhance.Contrast(ret_image)
                ret_image = contrast_image.enhance(factor=contrast)
            if saturation != 1:
                color_image = ImageEnhance.Color(ret_image)
                ret_image = color_image.enhance(factor=saturation)

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])
            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerColor: Brightness & Contrast": ColorCorrectBrightnessAndContrast,
    "LayerColor: BrightnessContrastV2": LS_ColorCorrect_Brightness_And_Contrast_V2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: BrightnessContrastV2": "LayerColor: Brightness Contrast V2"
}