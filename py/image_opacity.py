from .imagefunc import *

class ImageOpacity:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = 'image_opacity'
    CATEGORY = '😺dzNodes'
    OUTPUT_NODE = True

    def image_opacity(self, image, opacity, invert_mask,
                      mask=None,
                      ):

        _image = tensor2pil(image).convert('RGB')
        _mask = tensor2pil(image).convert('RGBA').split()[-1]
        if mask is not None:
            _mask = mask2image(mask).convert('L')

        # 设置透明度
        if invert_mask:
            _color = Image.new("L", _image.size, color=(255))
        else:
            _color = Image.new("L", _image.size, color=(0))
        # opacity
        ret_mask = _mask
        if opacity == 0:
            ret_mask = _color
        elif opacity < 100:
            alpha = 1.0 - float(opacity) / 100
            ret_mask = Image.blend(_mask, _color, alpha)
        R, G, B, = _image.split()
        if invert_mask:
            ret_image = Image.merge('RGBA', (R, G, B, ImageChops.invert(ret_mask)))
        else:
            ret_image = Image.merge('RGBA', (R, G, B, ret_mask))

        return (pil2tensor(ret_image), image2mask(ret_mask),)

NODE_CLASS_MAPPINGS = {
    "LayerStyle_ImageOpacity": ImageOpacity
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle_ImageOpacity": "LayerStyle: ImageOpacity"
}