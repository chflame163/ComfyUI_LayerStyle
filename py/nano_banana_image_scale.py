import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, fit_resize_image


PREFERED_BANANA_RESOLUTIONS = [
    (1024, 1024),
    (1184, 864),
    (864, 1184),
    (832, 1248),
    (1248, 832),
    (768, 1344),
    (1344, 768),
]


class LS_NanoBananaImageScale:
    @classmethod
    def INPUT_TYPES(s):
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        return {"required": {"image": ("IMAGE", ),
                             "method": (method_mode,),
                            },
               }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale"

    CATEGORY = '😺dzNodes/LayerUtility'
    DESCRIPTION = "This node resizes the image to one that is more optimal for nano-banana. For images with different aspect ratio, the scale will be adjusted appropriately to maintain all information"

    def scale(self, image, method):
        ret_images = []

        width = image.shape[2]
        height = image.shape[1]
        aspect_ratio = width / height
        _, target_width, target_height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_BANANA_RESOLUTIONS)

        resize_sampler = Image.LANCZOS
        if method == "bicubic":
            resize_sampler = Image.BICUBIC
        elif method == "hamming":
            resize_sampler = Image.HAMMING
        elif method == "bilinear":
            resize_sampler = Image.BILINEAR
        elif method == "box":
            resize_sampler = Image.BOX
        elif method == "nearest":
            resize_sampler = Image.NEAREST

        for img in  image:
            _image = torch.unsqueeze(img, 0)
            _image = tensor2pil(img).convert('RGB')
            resized_image = fit_resize_image(_image, target_width, target_height, 'fill', resize_sampler)
            ret_images.append(pil2tensor(resized_image))

        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: NanoBananaImageScale": LS_NanoBananaImageScale
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: NanoBananaImageScale": "LayerUtility: Nano Banana Image Scale"
}