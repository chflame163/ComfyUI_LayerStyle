from .imagefunc import *

any = AnyType("*")

class ImageMaskScaleAs:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        fit_mode = ['letterbox', 'crop', 'fill']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']

        return {
            "required": {
                "scale_as": (any, {}),
                "fit": (fit_mode,),
                "method": (method_mode,),
            },
            "optional": {
                "image": ("IMAGE",),  #
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX",)
    RETURN_NAMES = ("image", "mask", "original_size")
    FUNCTION = 'image_mask_scale_as'
    CATEGORY = '😺dzNodes/LayerUtility'
    OUTPUT_NODE = True

    def image_mask_scale_as(self, scale_as, fit, method,
                            image=None, mask = None,
                            ):

        _asimage = tensor2pil(scale_as)
        target_width, target_height = _asimage.size
        _mask = Image.new('L', size=_asimage.size, color='black')
        _image = Image.new('RGB', size=_asimage.size, color='black')
        orig_width = 4
        orig_height = 4
        if mask is not None:
            _mask = tensor2pil(mask).convert('L')
            orig_width, orig_height = _mask.size
        if image is not None:
            _image = tensor2pil(image).convert('RGB')
            orig_width, orig_height = _image.size
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
        if image is not None:
            _image = fit_resize_image(_image, target_width, target_height, fit, resize_sampler)
        if mask is not None:
            _mask = fit_resize_image(_mask, target_width, target_height, fit, resize_sampler).convert('L')

        return (pil2tensor(_image), image2mask(_mask), [orig_width, orig_height],)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageMaskScaleAs": ImageMaskScaleAs
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageMaskScaleAs": "LayerUtility: ImageMaskScaleAs"
}