import copy
from .imagefunc import *

class ImageBlendAdvance:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        mirror_mode = ['None', 'horizontal', 'vertical']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "x_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
                "y_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
                "mirror": (mirror_mode,),  # 镜像翻转
                "scale": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "aspect_ratio": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "rotate": ("FLOAT", {"default": 0, "min": -999999, "max": 999999, "step": 0.01}),
                "transform_method": (method_mode,),
                "anti_aliasing": ("INT", {"default": 0, "min": 0, "max": 16, "step": 0.01}),
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'image_blend_advance'
    CATEGORY = '😺dzNodes/LayerUtility'
    OUTPUT_NODE = True

    def image_blend_advance(self, background_image, layer_image,
                            invert_mask, blend_mode, opacity,
                            x_percent, y_percent,
                            mirror, scale, aspect_ratio, rotate,
                            transform_method, anti_aliasing,
                            layer_mask=None
                            ):

        # preprocess
        _canvas = tensor2pil(background_image).convert('RGB')
        _layer = tensor2pil(layer_image)
        if _layer.mode == 'RGBA':
            _mask = tensor2pil(layer_image).convert('RGBA').split()[-1]
        else:
            _mask = Image.new('L', _layer.size, 'white')
        _layer = _layer.convert('RGB')
        if layer_mask is not None:
            if invert_mask:
                layer_mask = 1 - layer_mask
            _mask = mask2image(layer_mask).convert('L')
        if _mask.size != _layer.size:
            _mask = Image.new('L', _layer.size, 'white')
            log('Warning: mask mismatch, droped!')

        orig_layer_width = _layer.width
        orig_layer_height = _layer.height
        _mask = _mask.convert("RGB")

        target_layer_width = int(orig_layer_width * scale)
        target_layer_height = int(orig_layer_height * scale * aspect_ratio)

        # mirror
        if mirror == 'horizontal':
            _layer = _layer.transpose(Image.FLIP_LEFT_RIGHT)
            _mask = _mask.transpose(Image.FLIP_LEFT_RIGHT)
        elif mirror == 'vertical':
            _layer = _layer.transpose(Image.FLIP_TOP_BOTTOM)
            _mask = _mask.transpose(Image.FLIP_TOP_BOTTOM)

        # scale
        _layer = _layer.resize((target_layer_width, target_layer_height))
        _mask = _mask.resize((target_layer_width, target_layer_height))
        # rotate
        _layer, _mask, _ = image_rotate_extend_with_alpha(_layer, rotate, _mask, transform_method, anti_aliasing)

        # 处理位置
        x = int(_canvas.width * x_percent / 100 - _layer.width / 2)
        y = int(_canvas.height * y_percent / 100 - _layer.height / 2)

        # composit layer
        _comp = copy.copy(_canvas)
        _compmask = Image.new("RGB", _comp.size, color='black')
        _comp.paste(_layer, (x, y))
        _compmask.paste(_mask, (x, y))
        _compmask = _compmask.convert('L')
        _comp = chop_image(_canvas, _comp, blend_mode, opacity)

        # composition background
        _canvas.paste(_comp, mask=_compmask)
        ret_image = _canvas
        ret_mask = image2mask(_compmask)
        log('ImageBlendAdvance Processed.')
        return (pil2tensor(ret_image), ret_mask,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageBlendAdvance": ImageBlendAdvance
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageBlendAdvance": "LayerUtility: ImageBlendAdvance"
}