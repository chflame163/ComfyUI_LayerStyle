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

        b_images = []
        l_images = []
        l_masks = []
        ret_images = []

        for b in background_image:
            b_images.append(b)
        for l in layer_image:
            l_images.append(l)
            m = tensor2pil(l)
            if tensor2pil(l).mode == 'RGBA':
                l_masks.append(m.convert('RGBA').split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))
        if layer_mask is not None:
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(m).convert('L'))
        max_batch = max(len(b_images), len(l_images), len(l_masks))

        _color = Image.new("RGB", tensor2pil(l_images[0]).size, color=color)
        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            layer_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            # preprocess
            _canvas = tensor2pil(background_image).convert('RGB')
            _layer = tensor2pil(layer_image).convert('RGB')

            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                log('Warning: mask mismatch, droped!')

            # 合成layer
            _comp = chop_image(_layer, _color, blend_mode, opacity)
            _canvas.paste(_comp, mask=_mask)

            ret_images.append(pil2tensor(_canvas))

        log(f'ColorOverlay Processed {len(ret_images)} image(s).')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerStyle: ColorOverlay": ColorOverlay
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle: ColorOverlay": "LayerStyle: ColorOverlay"
}