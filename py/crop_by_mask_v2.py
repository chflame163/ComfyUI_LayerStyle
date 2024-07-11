from .imagefunc import *

NODE_NAME = 'CropByMask V2'

class CropByMaskV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['mask_area', 'min_bounding_rect', 'max_inscribed_rect']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "detect": (detect_mode,),
                "top_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "round_to_multiple": (multiple_list,),
            },
            "optional": {
                "crop_box": ("BOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "IMAGE",)
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_box", "box_preview")
    FUNCTION = 'crop_by_mask_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def crop_by_mask_v2(self, image, mask, invert_mask, detect,
                     top_reserve, bottom_reserve,
                     left_reserve, right_reserve, round_to_multiple,
                     crop_box=None
                     ):

        ret_images = []
        ret_masks = []
        l_images = []
        l_masks = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        # 如果有多张mask输入，使用第一张
        if mask.shape[0] > 1:
            log(f"Warning: Multiple mask inputs, using the first.", message_type='warning')
            mask = torch.unsqueeze(mask[0], 0)
        if invert_mask:
            mask = 1 - mask
        l_masks.append(tensor2pil(torch.unsqueeze(mask, 0)).convert('L'))

        _mask = mask2image(mask)
        preview_image = tensor2pil(mask).convert('RGB')
        if crop_box is None:
            bluredmask = gaussian_blur(_mask, 20).convert('L')
            x = 0
            y = 0
            width = 0
            height = 0
            if detect == "min_bounding_rect":
                (x, y, w, h) = min_bounding_rect(bluredmask)
            elif detect == "max_inscribed_rect":
                (x, y, w, h) = max_inscribed_rect(bluredmask)
            else:
                (x, y, w, h) = mask_area(_mask)

            canvas_width, canvas_height = tensor2pil(torch.unsqueeze(image[0], 0)).convert('RGB').size
            x1 = x - left_reserve if x - left_reserve > 0 else 0
            y1 = y - top_reserve if y - top_reserve > 0 else 0
            x2 = x + w + right_reserve if x + w + right_reserve < canvas_width else canvas_width
            y2 = y + h + bottom_reserve if y + h + bottom_reserve < canvas_height else canvas_height

            if round_to_multiple != 'None':
                multiple = int(round_to_multiple)
                width = num_round_up_to_multiple(x2 - x1, multiple)
                height = num_round_up_to_multiple(y2 - y1, multiple)
                x1 = x1 - (width - (x2 - x1)) // 2
                y1 = y1 - (height - (y2 - y1)) // 2
                x2 = x1 + width
                y2 = y1 + height

            log(f"{NODE_NAME}: Box detected. x={x1},y={y1},width={width},height={height}")
            crop_box = (x1, y1, x2, y2)
            preview_image = draw_rect(preview_image, x, y, w, h, line_color="#F00000",
                                      line_width=(w + h) // 100)
        preview_image = draw_rect(preview_image, crop_box[0], crop_box[1],
                                  crop_box[2] - crop_box[0], crop_box[3] - crop_box[1],
                                  line_color="#00F000",
                                  line_width=(crop_box[2] - crop_box[0] + crop_box[3] - crop_box[1]) // 200)
        for i in range(len(l_images)):
            _canvas = tensor2pil(l_images[i]).convert('RGB')
            _mask = l_masks[0]
            ret_images.append(pil2tensor(_canvas.crop(crop_box)))
            ret_masks.append(image2mask(_mask.crop(crop_box)))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0), list(crop_box), pil2tensor(preview_image),)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: CropByMask V2": CropByMaskV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: CropByMask V2": "LayerUtility: CropByMask V2"
}