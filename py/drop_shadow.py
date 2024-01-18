from .imagefunc import *

class DropShadow:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        chop_mode = ['normal','multply','screen','add','subtract','difference','darker','lighter']
        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode,),  # 混合模式
                "opacity": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),  # 透明度
                "distance_x": ("INT", {"default": 5, "min": -9999, "max": 9999, "step": 1}),  # x_偏移
                "distance_y": ("INT", {"default": 5, "min": -9999, "max": 9999, "step": 1}),  # y_偏移
                "grow": ("INT", {"default": 2, "min": -9999, "max": 9999, "step": 1}),  # 扩张
                "blur": ("INT", {"default": 15, "min": 0, "max": 100, "step": 1}),  # 模糊
                "shadow_color": ("STRING", {"default": "#000000"}),  # 背景颜色
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    # RETURN_TYPES = ("IMAGE", "MASK",)
    # RETURN_NAMES = ("image", "shadow_mask",)
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'drop_shadow'
    CATEGORY = '😺dzNodes'
    OUTPUT_NODE = True

    def drop_shadow(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity, distance_x, distance_y,
                  grow, blur, shadow_color,
                  layer_mask=None,
                  ):
        distance_x = -distance_x
        distance_y = -distance_y
        _canvas = tensor2pil(background_image).convert('RGB')
        _layer = tensor2pil(layer_image).convert('RGB')
        _mask = tensor2pil(layer_image).convert('RGBA').split()[-1]
        # 处理mask
        if layer_mask is not None:
            if invert_mask:
                layer_mask = 1 - layer_mask
            _mask = mask2image(layer_mask).convert('L')

        if distance_x != 0 or distance_y != 0:
            __mask = shift_image(_mask, distance_x, distance_y)  # 位移
        shadow_mask = expand_mask(image2mask(__mask), grow, blur, 0)  #扩张，模糊

        # 合成阴影
        shadow_color = Image.new("RGB", _layer.size, color=shadow_color)
        alpha = tensor2pil(shadow_mask).convert('L')
        _shadow = chop_image(_canvas, shadow_color, blend_mode, opacity)
        _canvas.paste(_shadow, mask=alpha)

        # 合成layer
        _canvas.paste(_layer, mask=_mask)

        ret_image = _canvas
        # ret_mask = shadow_mask
        # return (pil2tensor(ret_image), ret_mask,)
        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerStyle_DropShadow": DropShadow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle_DropShadow": "LayerStyle: DropShadow"
}