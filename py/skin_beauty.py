from .imagefunc import *

class SkinBeauty:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
                "smooth": ("INT", {"default": 20, "min": 1, "max": 64, "step": 1}),  # 磨皮程度
                "threshold": ("INT", {"default": -10, "min": -255, "max": 255, "step": 1}),  # 高光阈值
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "beauty_mask")
    FUNCTION = 'skin_beauty'
    CATEGORY = '😺dzNodes/LayerFilter'
    OUTPUT_NODE = True

    def skin_beauty(self, image, smooth, threshold, opacity
                  ):

        _canvas = tensor2pil(image).convert('RGB')
        _R, _, _, _ = image_channel_split(_canvas, mode='RGB')
        _otsumask = gray_threshold(_R, otsu=True)
        _removebkgd = remove_background(_R, _otsumask, '#000000')
        auto_threshold = get_image_bright_average(_removebkgd) - 16
        light_mask = gray_threshold(_canvas, auto_threshold + threshold)
        blur = int((_canvas.width + _canvas.height) / 2000 * smooth)
        _image = image_beauty(_canvas, level=smooth)
        _image = gaussian_blur(_image, blur)
        _image = chop_image(_canvas, _image, 'normal', opacity)
        _canvas.paste(_image, mask=gaussian_blur(light_mask, blur).convert('L'))

        return (pil2tensor(_canvas), image2mask(light_mask),)

NODE_CLASS_MAPPINGS = {
    "LayerFilter: SkinBeauty": SkinBeauty
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: SkinBeauty": "LayerFilter: SkinBeauty"
}