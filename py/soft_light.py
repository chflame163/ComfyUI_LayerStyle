from .imagefunc import *

NODE_NAME = 'SoftLight'

class SoftLight:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "soft": ("FLOAT", {"default": 1, "min": 0.2, "max": 10, "step": 0.01}),  # 模糊
                "threshold": ("INT", {"default": -10, "min": -255, "max": 255, "step": 1}),  # 高光阈值
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'soft_light'
    CATEGORY = '😺dzNodes/LayerFilter'

    def soft_light(self, image, soft, threshold, opacity,):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            blend_mode = 'screen'
            _canvas = tensor2pil(i).convert('RGB')
            blur = int((_canvas.width + _canvas.height) / 200 * soft)
            _otsumask = gray_threshold(_canvas, otsu=True)
            _removebkgd = remove_background(_canvas, _otsumask, '#000000').convert('L')
            auto_threshold = get_image_bright_average(_removebkgd)
            light_mask = gray_threshold(_canvas, auto_threshold + threshold)
            highlight_mask = gray_threshold(_canvas, auto_threshold + (255 - auto_threshold) // 2 + threshold // 2)
            blurimage = gaussian_blur(_canvas, soft).convert('RGB')
            light = chop_image(_canvas, blurimage, blend_mode=blend_mode, opacity=opacity)
            highlight = chop_image(light, blurimage, blend_mode=blend_mode, opacity=opacity)
            _canvas.paste(highlight, mask=gaussian_blur(light_mask, blur * 2).convert('L'))
            _canvas.paste(highlight, mask=gaussian_blur(highlight_mask, blur).convert('L'))

            ret_images.append(pil2tensor(_canvas))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerFilter: SoftLight": SoftLight
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: SoftLight": "LayerFilter: SoftLight"
}