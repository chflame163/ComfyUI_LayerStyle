from .imagefunc import *

class ImageBlend:

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
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'image_blend'
    CATEGORY = '😺dzNodes/LayerUtility'
    OUTPUT_NODE = True

    def image_blend(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity,
                  layer_mask=None
                  ):

        _canvas = tensor2pil(background_image).convert('RGB')
        _layer = tensor2pil(layer_image).convert('RGB')
        _mask = tensor2pil(layer_image).convert('RGBA').split()[-1]
        if layer_mask is not None:
            if invert_mask:
                layer_mask = 1 - layer_mask
            _mask = mask2image(layer_mask).convert('L')
        # 合成layer
        _comp = chop_image(_canvas, _layer, blend_mode, opacity)
        _canvas.paste(_comp, mask=_mask)
        ret_image = _canvas
        log('ImageBlend Processed.')
        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageBlend": ImageBlend
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageBlend": "LayerUtility: ImageBlend"
}