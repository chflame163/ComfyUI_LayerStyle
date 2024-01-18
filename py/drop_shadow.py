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
                "layer_mask": ("MASK",),  #
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
                # "test_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "shadow_mask",)
    FUNCTION = 'drop_shadow'
    CATEGORY = '😺dzNodes'
    OUTPUT_NODE = True

    def drop_shadow(self, background_image, layer_image, layer_mask,
                  invert_mask, blend_mode, opacity, distance_x, distance_y,
                  grow, blur, shadow_color,
                  ):
        distance_x = -distance_x
        distance_y = -distance_y
        # 处理阴影mask
        if invert_mask:
            layer_mask = 1 - layer_mask
        _layer = tensor2pil(layer_image)
        _mask = mask2image(layer_mask)
        if distance_x != 0 or distance_y != 0:
            _mask = shift_image(_mask, distance_x, distance_y)  # 位移
        shadow_mask = expand_mask(image2mask(_mask), grow, blur, 0)  #扩张，模糊

        # 合成阴影
        shadow_color = Image.new("RGB", _layer.size, color=shadow_color)
        alpha = tensor2pil(shadow_mask).convert('L')
        _canvas = tensor2pil(background_image)
        _shadow = chop_image(tensor2pil(background_image), shadow_color, blend_mode, opacity)
        _canvas.paste(_shadow, mask=alpha)

        # 合成layer
        alpha = tensor2pil(layer_mask).convert('L')
        _canvas.paste(_layer, mask=alpha)

        ret_image = _canvas
        ret_mask = shadow_mask
        return (pil2tensor(ret_image), ret_mask,)

NODE_CLASS_MAPPINGS = {
    "LayerStyle_DropShadow": DropShadow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle_DropShadow": "Layer Style: Drop Shadow"
}