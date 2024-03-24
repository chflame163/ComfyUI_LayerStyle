from .imagefunc import *

NODE_NAME = 'ImageRemoveAlpha'

class ImageRemoveAlpha:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "RGBA_image": ("IMAGE", ),  #
                "fill_background": ("BOOLEAN", {"default": False}),
                "background_color": ("STRING", {"default": "#000000"}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("RGB_image", )
    FUNCTION = 'image_remove_alpha'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_remove_alpha(self, RGBA_image, fill_background, background_color):

        ret_images = []

        for i in RGBA_image:
            i = torch.unsqueeze(i, 0)
            _image = tensor2pil(i)
            if _image.mode != "RGBA":
                log(f"Error: {NODE_NAME} skipped, because the input image is not RGBA.", message_type='error')
                return (RGBA_image)
            if fill_background:
                alpha = _image.split()[-1]
                ret_image = Image.new('RGB', size=_image.size, color=background_color)
                ret_image.paste(_image, mask=alpha)
                ret_images.append(pil2tensor(ret_image))
            else:
                ret_images.append(pil2tensor(tensor2pil(i).convert('RGB')))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), )

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageRemoveAlpha": ImageRemoveAlpha
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageRemoveAlpha": "LayerUtility: ImageRemoveAlpha"
}