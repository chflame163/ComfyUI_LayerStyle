from .imagefunc import *

class OuterGlow:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        chop_mode = ['normal','multply','screen','add','subtract','difference','darker','lighter']
        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "layer_mask": ("MASK",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "grow": ("INT", {"default": 16, "min": -9999, "max": 9999, "step": 1}),  # 扩张
                "blur": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),  # 模糊
                "glow_color": ("STRING", {"default": "#FFFFFF"}),  # 背景颜色
            },
            "optional": {
            }
        }

    # RETURN_TYPES = ("IMAGE", "MASK",)
    # RETURN_NAMES = ("image", "glow_mask",)
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'outer_glow'
    CATEGORY = '😺dzNodes'
    OUTPUT_NODE = True

    def outer_glow(self, background_image, layer_image, layer_mask,
                  invert_mask, blend_mode, opacity,
                  grow, blur, glow_color,
                  ):

        # 处理mask
        if invert_mask:
            layer_mask = 1 - layer_mask
        _layer = tensor2pil(layer_image)
        _mask = mask2image(layer_mask)
        glow_mask = expand_mask(image2mask(_mask), grow, blur, 0)  #扩张，模糊

        # 合成glow
        glow_color = Image.new("RGB", _layer.size, color=glow_color)
        alpha = tensor2pil(glow_mask).convert('L')
        _canvas = tensor2pil(background_image)
        _glow = chop_image(tensor2pil(background_image), glow_color, blend_mode, opacity)
        _canvas.paste(_glow, mask=alpha)

        # 合成layer
        alpha = tensor2pil(layer_mask).convert('L')
        _canvas.paste(_layer, mask=alpha)

        ret_image = _canvas
        ret_mask = glow_mask
        # return (pil2tensor(ret_image), ret_mask,)
        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerStyle_OuterGlow": OuterGlow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle_OuterGlow": "LayerStyle: OuterGlow"
}