import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, image2mask, gaussian_blur, chop_image
from .imagefunc import image_channel_split, gray_threshold, remove_background, get_image_bright_average, image_beauty


class SkinBeauty:

    def __init__(self):
        self.NODE_NAME = 'SkinBeauty'

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

    def skin_beauty(self, image, smooth, threshold, opacity
                  ):

        ret_images = []
        ret_masks = []
        for i in image:
            i = torch.unsqueeze(i, 0)
            _canvas = tensor2pil(i).convert('RGB')
            _R, _, _, _ = image_channel_split(_canvas, mode='RGB')
            _otsumask = gray_threshold(_R, otsu=True)
            _removebkgd = remove_background(_R, _otsumask, '#000000')
            auto_threshold = get_image_bright_average(_removebkgd) - 16
            light_mask = gray_threshold(_canvas, auto_threshold + threshold)
            blur = int((_canvas.width + _canvas.height) / 2000 * smooth)
            _image = image_beauty(_canvas, level=smooth)
            _image = gaussian_blur(_image, blur)
            _image = chop_image(_canvas, _image, 'normal', opacity)
            light_mask = gaussian_blur(light_mask, blur).convert('L')
            _canvas.paste(_image, mask=light_mask)

            ret_images.append(pil2tensor(_canvas))
            ret_masks.append(image2mask(light_mask))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerFilter: SkinBeauty": SkinBeauty
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: SkinBeauty": "LayerFilter: SkinBeauty"
}