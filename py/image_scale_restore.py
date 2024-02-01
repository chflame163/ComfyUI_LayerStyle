from .imagefunc import *

class ImageScaleRestore:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "scale": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "method": (method_mode,),
                "scale_by_longest_side": ("BOOLEAN", {"default": False}),  # 是否按长边缩放
                "longest_side": ("INT", {"default": 1024, "min": 4, "max": 999999, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),  #
                "original_size": ("BOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX",)
    RETURN_NAMES = ("image", "mask", "original_size")
    FUNCTION = 'image_scale_restore'
    CATEGORY = '😺dzNodes/LayerUtility'
    OUTPUT_NODE = True

    def image_scale_restore(self, image, scale, method,
                            scale_by_longest_side, longest_side,
                            mask = None,  original_size = None
                            ):

        _canvas = tensor2pil(image).convert('RGB')
        orig_width = _canvas.width
        orig_height = _canvas.height
        if original_size is not None:
            target_width = original_size[0]
            target_height = original_size[1]
        else:
            target_width = int(orig_width * scale)
            target_height = int(orig_height * scale)
            if scale_by_longest_side:
                if orig_width > orig_height:
                    target_width = longest_side
                    target_height = int(target_width * orig_height / orig_width)
                else:
                    target_height = longest_side
                    target_width = int(target_height * orig_width / orig_height)
        if target_width < 4:
            target_width = 4
        if target_height < 4:
            target_height = 4
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
        ret_image = _canvas.resize((target_width, target_height), resize_sampler)
        ret_mask = Image.new('L', size=ret_image.size, color='white')
        if mask is not None:
            _mask = mask2image(mask).convert('L')
            ret_mask = _mask.resize((target_width, target_height), resize_sampler)

        return (pil2tensor(ret_image), image2mask(ret_mask), [orig_width, orig_height],)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageScaleRestore": ImageScaleRestore
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageScaleRestore": "LayerUtility: ImageScaleRestore"
}