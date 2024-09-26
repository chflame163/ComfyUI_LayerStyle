from .imagefunc import *

NODE_NAME = 'GaussianBlur'

class GaussianBlur:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "blur": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),  # 模糊
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'gaussian_blur'
    CATEGORY = '😺dzNodes/LayerFilter'

    def gaussian_blur(self, image, blur):

        ret_images = []

        for i in image:
            _canvas = tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')

            ret_images.append(pil2tensor(gaussian_blur(_canvas, blur)))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)


class LS_GaussianBlurV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "blur": ("FLOAT", {"default": 20, "min": 0, "max": 1000, "step": 0.05}),  # 模糊
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'gaussian_blur_v2'
    CATEGORY = '😺dzNodes/LayerFilter'

    def gaussian_blur_v2(self, image, blur):

        ret_images = []

        if blur:
            for i in image:
                _canvas = tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')

                ret_images.append(pil2tensor(gaussian_blur(_canvas, blur)))
        else:
            return (image,)

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerFilter: GaussianBlur": GaussianBlur,
    "LayerFilter: GaussianBlurV2": LS_GaussianBlurV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: GaussianBlur": "LayerFilter: GaussianBlur",
    "LayerFilter: GaussianBlurV2": "LayerFilter: Gaussian Blur V2"
}