from .imagefunc import *

NODE_NAME = 'AutoAdjustV2'

class AutoAdjustV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        mode_list = ["RGB", "lum + sat", "mono", "luminance", "saturation"]
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "strength": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "brightness": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "contrast": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "saturation": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "red": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "green": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "blue": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "mode": (mode_list, ),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'auto_adjust_v2'
    CATEGORY = '😺dzNodes/LayerColor'


    def auto_adjust_v2(self, image, strength, brightness, contrast, saturation, red, green, blue, mode, mask=None):

        def auto_level_gray(image, mask):
            gray_image = Image.new("L", image.size, color='gray')
            gray_image.paste(image.convert('L'), mask=mask)
            return normalize_gray(gray_image)

        if brightness < 0:
            brightness_offset = brightness / 100 + 1
        else:
            brightness_offset = brightness / 50 + 1
        if contrast < 0:
            contrast_offset = contrast / 100 + 1
        else:
            contrast_offset = contrast / 50 + 1
        if saturation < 0:
            saturation_offset = saturation / 100 + 1
        else:
            saturation_offset = saturation / 50 + 1

        l_images = []
        l_masks = []
        ret_images = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        max_batch = max(len(l_images), len(l_masks))
        for i in range(max_batch):
            _image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            orig_image = tensor2pil(_image)

            if mode == 'RGB':
                r, g, b, _  = image_channel_split(orig_image, mode = 'RGB')
                r = auto_level_gray(r, _mask)
                g = auto_level_gray(g, _mask)
                b = auto_level_gray(b, _mask)
                ret_image = image_channel_merge((r, g, b), 'RGB')
            elif mode == 'lum + sat':
                h, s, v, _ = image_channel_split(orig_image, mode = 'HSV')
                s = auto_level_gray(s, _mask)
                ret_image = image_channel_merge((h, s, v), 'HSV')
                l, a, b, _ = image_channel_split(ret_image, mode = 'LAB')
                l = auto_level_gray(l, _mask)
                ret_image = image_channel_merge((l, a, b), 'LAB')
            elif mode == 'luminance':
                l, a, b, _ = image_channel_split(orig_image, mode = 'LAB')
                l = auto_level_gray(l, _mask)
                ret_image = image_channel_merge((l, a, b), 'LAB')
            elif mode == 'saturation':
                h, s, v, _ = image_channel_split(orig_image, mode = 'HSV')
                s = auto_level_gray(s, _mask)
                ret_image = image_channel_merge((h, s, v), 'HSV')
            else: # mono
                gray = orig_image.convert('L')
                ret_image = auto_level_gray(gray, _mask).convert('RGB')

            if (red or green or blue) and mode != "mono":
                r, g, b, _ = image_channel_split(ret_image, mode='RGB')
                if red:
                    r = gamma_trans(r, self.balance_to_gamma(red)).convert('L')
                if green:
                    g = gamma_trans(g, self.balance_to_gamma(green)).convert('L')
                if blue:
                    b = gamma_trans(b, self.balance_to_gamma(blue)).convert('L')
                ret_image = image_channel_merge((r, g, b), 'RGB')

            if brightness:
                brightness_image = ImageEnhance.Brightness(ret_image)
                ret_image = brightness_image.enhance(factor=brightness_offset)
            if contrast:
                contrast_image = ImageEnhance.Contrast(ret_image)
                ret_image = contrast_image.enhance(factor=contrast_offset)
            if saturation:
                color_image = ImageEnhance.Color(ret_image)
                ret_image = color_image.enhance(factor=saturation_offset)
            ret_image = chop_image_v2(orig_image, ret_image, blend_mode="normal", opacity=strength)
            ret_image.paste(orig_image, mask=ImageChops.invert(_mask))
            if orig_image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, orig_image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

    def balance_to_gamma(self, balance:int) -> float:
        return 0.00005 * balance * balance - 0.01 * balance + 1

NODE_CLASS_MAPPINGS = {
    "LayerColor: AutoAdjustV2": AutoAdjustV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: AutoAdjustV2": "LayerColor: AutoAdjust V2"
}