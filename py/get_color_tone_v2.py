from .imagefunc import *

NODE_NAME = 'GetColorToneV2'

class GetColorToneV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        remove_background_list = ['none','BiRefNet', 'RMBG 1.4',]
        subject_list = ['mask','entire', 'background', 'subject']
        mode_list = ['main_color', 'average']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mode": (mode_list,),  # 主色/平均色
                "color_of": (subject_list,),
                "remove_bkgd_method": (remove_background_list,),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "mask_grow": ("INT", {"default": 0, "min": -999, "max": 999, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "LIST", "MASK")
    RETURN_NAMES = ("image", "color_in_hex", "HSV color in list", "mask",)
    FUNCTION = 'get_color_tone_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def get_color_tone_v2(self, image, mode, remove_bkgd_method, color_of, invert_mask, mask_grow,
                  mask=None
                  ):

        _images = []
        _masks = []
        ret_images = []
        ret_masks = []
        need_rmbg = False
        for i in image:
            _images.append(torch.unsqueeze(i, 0))
            m = tensor2pil(i)
            if m.mode == 'RGBA':
                _masks.append(1 - image2mask(m.split()[-1]))
            else:
                _masks.append(pil2tensor(Image.new("L", (m.width, m.height), color="white")))
                if remove_bkgd_method != 'none':
                    need_rmbg = True

        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            _masks = []
            for m in mask:
                _masks.append(torch.unsqueeze(m, 0))
            need_rmbg = False

        max_batch = max(len(_images), len(_masks))

        if remove_bkgd_method == 'BiRefNet':
            from .birefnet_legacy import BiRefNetRemoveBackground
            birefnetrmbg = BiRefNetRemoveBackground()

        for i in range(max_batch):
            _image = _images[i] if i < len(_images) else _images[-1]
            _image = tensor2pil(_image).convert("RGB")
            if need_rmbg:
                if remove_bkgd_method == 'BiRefNet':
                    _mask = birefnetrmbg.generate_mask(_image)
                else:
                    _mask = RMBG(_image)
                _mask = image2mask(_mask)
            else:
                _mask = _masks[i] if i < len(_masks) else _masks[-1]

            if invert_mask:
                _mask = 1 - _mask

            if mask_grow != 0:
                _mask = expand_mask(_mask, mask_grow, 0)  # 扩张，模糊

            if color_of == 'entire':
                blured_image = gaussian_blur(_image, int((_image.width + _image.height) / 400))
            else:
                if color_of == 'background':
                    _mask = 1 - _mask
                _mask = tensor2pil(_mask)
                pixel_spread_image = pixel_spread(_image, _mask.convert('RGB'))
                blured_image = gaussian_blur(pixel_spread_image, int((_image.width + _image.height) / 400))

            ret_color = '#000000'
            if mode == 'main_color' and color_of != 'mask':
                ret_color = get_image_color_tone(blured_image)
            elif mode == 'average' and color_of != 'mask':
                ret_color = get_image_color_average(blured_image)
            elif mode == 'main_color' and color_of == 'mask':
                ret_color = get_image_color_tone(blured_image, mask=_mask)
            elif mode == 'average' and color_of == 'mask':
                ret_color = get_image_color_average(blured_image, mask=_mask)

            ret_image = Image.new('RGB', size=_image.size, color=ret_color)
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(pil2tensor(_mask))
        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        hsv_color = RGB_to_HSV(Hex_to_RGB(ret_color))
        return (torch.cat(ret_images, dim=0), ret_color, hsv_color, torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: GetColorToneV2": GetColorToneV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: GetColorToneV2": "LayerUtility: GetColorTone V2"
}