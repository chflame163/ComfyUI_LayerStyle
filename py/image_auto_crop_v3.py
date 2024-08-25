from .imagefunc import *

NODE_NAME = 'ImageAutoCropV3'

class ImageAutoCropV3:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        ratio_list = ['1:1', '3:2', '4:3', '16:9', '2:3', '3:4', '9:16', 'custom', 'original']
        scale_to_side_list = ['None', 'longest', 'shortest', 'width', 'height', 'total_pixel(kilo pixel)']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        return {
            "required": {
                "image": ("IMAGE", ),
                "aspect_ratio": (ratio_list,),
                "proportional_width": ("INT", {"default": 1, "min": 1, "max": 99999999, "step": 1}),
                "proportional_height": ("INT", {"default": 1, "min": 1, "max": 99999999, "step": 1}),
                "method": (method_mode,),
                "scale_to_side": (scale_to_side_list,),
                "scale_to_length": ("INT", {"default": 1024, "min": 4, "max": 999999, "step": 1}),
                "round_to_multiple": (multiple_list,),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("cropped_image", "box_preview",)
    FUNCTION = 'image_auto_crop_v3'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_auto_crop_v3(self, image, aspect_ratio,
                        proportional_width, proportional_height, method,
                        scale_to_side, scale_to_length, round_to_multiple,
                        mask=None,
                        ):

        ret_images = []
        ret_box_previews = []
        ret_masks = []
        input_images = []
        input_masks = []
        crop_boxs = []

        for l in image:
            input_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                input_masks.append(m.split()[-1])
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            input_masks = []
            for m in mask:
                input_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        if len(input_masks) > 0 and len(input_masks) != len(input_images):
            input_masks = []
            log(f"Warning, {NODE_NAME} unable align alpha to image, drop it.", message_type='warning')

        fit = 'crop'
        _image = tensor2pil(input_images[0])
        (orig_width, orig_height) = _image.size
        if aspect_ratio == 'custom':
            ratio = proportional_width / proportional_height
        elif aspect_ratio == 'original':
            ratio = orig_width / orig_height
        else:
            s = aspect_ratio.split(":")
            ratio = int(s[0]) / int(s[1])

        resize_sampler = Image.LANCZOS
        if method == "bicubic":
            resize_sampler = Image.BICUBIC
        elif method == "hamming":
            resize_sampler = Image.HAMMING
        elif method == "bilinear":
            resize_sampler = Image.BILINEAR
        elif method == "box":
            resize_sampler = Image.BOX
        elif method == "nearest":
            resize_sampler = Image.NEAREST

        # calculate target width and height
        if ratio > 1:
            if scale_to_side == 'longest':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'shortest':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'width':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'height':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'total_pixel(kilo pixel)':
                target_width = math.sqrt(ratio * scale_to_length * 1000)
                target_height = target_width / ratio
                target_width = int(target_width)
                target_height = int(target_height)
            else:
                target_width = orig_width
                target_height = int(target_width / ratio)
        else:
            if scale_to_side == 'longest':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'shortest':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'width':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'height':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'total_pixel(kilo pixel)':
                target_width = math.sqrt(ratio * scale_to_length * 1000)
                target_height = target_width / ratio
                target_width = int(target_width)
                target_height = int(target_height)
            else:
                target_height = orig_height
                target_width = int(target_height * ratio)

        if round_to_multiple != 'None':
            multiple = int(round_to_multiple)
            target_width = num_round_up_to_multiple(target_width, multiple)
            target_height = num_round_up_to_multiple(target_height, multiple)

        for i in range(len(input_images)):
            _image = tensor2pil(input_images[i]).convert('RGB')

            if len(input_masks) > 0:
                _mask = input_masks[i]
            else:
                _mask = Image.new('L', _image.size, color='black')

            bluredmask = gaussian_blur(_mask, 20).convert('L')
            (mask_x, mask_y, mask_w, mask_h) = mask_area(bluredmask)
            orig_ratio = _image.width / _image.height
            target_ratio = target_width / target_height
            # crop image to target ratio
            if orig_ratio > target_ratio: # crop LiftRight side
                crop_w = int(_image.height * target_ratio)
                crop_h = _image.height
            else: # crop TopBottom side
                crop_w = _image.width
                crop_h = int(_image.width / target_ratio)
            crop_x = mask_w // 2 + mask_x - crop_w // 2
            if crop_x < 0:
                crop_x = 0
            if crop_x + crop_w > _image.width:
                crop_x = _image.width - crop_w
            crop_y = mask_h // 2 + mask_y - crop_h // 2
            if crop_y < 0:
                crop_y = 0
            if crop_y + crop_h > _image.height:
                crop_y = _image.height - crop_h
            crop_image = _image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
            line_width = (_image.width + _image.height) // 200
            preview_image = draw_rect(_image, crop_x, crop_y,
                                      crop_w, crop_h,
                                      line_color="#F00000", line_width=line_width)
            ret_image = crop_image.resize((target_width, target_height), resize_sampler)
            ret_images.append(pil2tensor(ret_image))
            ret_box_previews.append(pil2tensor(preview_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),
                torch.cat(ret_box_previews, dim=0),
                )

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageAutoCrop V3": ImageAutoCropV3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageAutoCrop V3": "LayerUtility: ImageAutoCrop V3"
}