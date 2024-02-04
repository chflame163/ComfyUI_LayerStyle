from .imagefunc import *

class ColorOverlay:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "color": ("STRING", {"default": "#FFBF30"}),  # 渐变开始颜色
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_overlay'
    CATEGORY = '😺dzNodes/LayerStyle'
    OUTPUT_NODE = True

    def color_overlay(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity, color,
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

        _color = Image.new('RGB', size=_layer.size, color=color)
        # 合成layer
        _comp = chop_image(_layer, _color, blend_mode, opacity)
        _canvas.paste(_comp, mask=_mask)
        ret_image = _canvas
        log('ColorOverlay Processed.')
        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerStyle: ColorOverlay": ColorOverlay
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle: ColorOverlay": "LayerStyle: ColorOverlay"
}