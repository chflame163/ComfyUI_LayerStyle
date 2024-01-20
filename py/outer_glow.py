from .imagefunc import *

class OuterGlow:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        chop_mode = ['screen', 'add', 'lighter', 'normal', 'multply', 'subtract','difference','darker']
        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "brightness": ("INT", {"default": 5, "min": 2, "max": 20, "step": 1}),  # 迭代
                "glow_range": ("INT", {"default": 48, "min": -9999, "max": 9999, "step": 1}),  # 扩张
                "blur": ("INT", {"default": 25, "min": 0, "max": 9999, "step": 1}),  # 扩张
                "light_color": ("STRING", {"default": "#FFBF30"}),  # 光源中心颜色
                "glow_color": ("STRING", {"default": "#FE0000"}),  # 辉光外围颜色
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'outer_glow'
    CATEGORY = '😺dzNodes'
    OUTPUT_NODE = True

    def outer_glow(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity,
                  brightness, glow_range, blur, light_color, glow_color,
                  layer_mask=None,
                  ):

        log('OuterGLow Advance Processing...')
        _canvas = tensor2pil(background_image).convert('RGB')
        _layer = tensor2pil(layer_image).convert('RGB')
        _mask = tensor2pil(layer_image).convert('RGBA').split()[-1]
        # 处理mask
        if layer_mask is not None:
            if invert_mask:
                layer_mask = 1 - layer_mask
            _mask = mask2image(layer_mask).convert('L')
        blur_factor = blur / 20.0
        grow = glow_range
        for x in range(brightness):
            blur = int(grow * blur_factor)
            _color = step_color(glow_color, light_color, brightness, x)
            glow_mask = expand_mask(image2mask(_mask), grow, blur)  #扩张，模糊
            # 合成glow
            color_image = Image.new("RGB", _layer.size, color=_color)
            alpha = tensor2pil(glow_mask).convert('L')
            _glow = chop_image(_canvas, color_image, blend_mode, step_value(1, opacity, brightness, x))
            _canvas.paste(_glow, mask=alpha)
            grow = grow - int(glow_range/brightness)

        # 合成layer
        _canvas.paste(_layer, mask=_mask)
        ret_image = _canvas

        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerStyle_OuterGlow": OuterGlow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle_OuterGlow": "LayerStyle: OuterGlow"
}