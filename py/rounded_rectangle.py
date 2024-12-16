import torch
from PIL import Image
from .imagefunc import log, pil2tensor, tensor2pil, image2mask, RGB2RGBA
from .imagefunc import draw_rounded_rectangle, gaussian_blur, mask_area, max_inscribed_rect, min_bounding_rect


class LS_RoundedRectangle:

    def __init__(self):
        self.NODE_NAME = 'RoundedRectangle'

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['mask_area', 'min_bounding_rect', 'max_inscribed_rect']
        return {
            "required": {
                "image": ("IMAGE",),
                "rounded_rect_radius": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "anti_aliasing": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
                "top": ("FLOAT", {"default": 8, "min": -100, "max": 100, "step": 0.1}),
                "bottom": ("FLOAT", {"default": 8, "min": -100, "max": 100, "step": 0.1}),
                "left": ("FLOAT", {"default": 8, "min": -100, "max": 100, "step": 0.1}),
                "right": ("FLOAT", {"default": 8, "min": -100, "max": 100, "step": 0.1}),
                "detect": (detect_mode,),
                "obj_ext_top": ("FLOAT", {"default": 8, "min": -100, "max": 100, "step": 0.1}),
                "obj_ext_bottom": ("FLOAT", {"default": 8, "min": -100, "max": 100, "step": 0.1}),
                "obj_ext_left": ("FLOAT", {"default": 8, "min": -100, "max": 100, "step": 0.1}),
                "obj_ext_right": ("FLOAT", {"default": 8, "min": -100, "max": 100, "step": 0.1}),
            },
            "optional": {
                "object_mask": ("MASK",),
                "crop_box": ("BOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = 'rounded_rectangle'
    CATEGORY = '😺dzNodes/LayerUtility'

    def rounded_rectangle(self, image, rounded_rect_radius, anti_aliasing, top, bottom, left, right,
                          detect, obj_ext_top, obj_ext_bottom, obj_ext_left, obj_ext_right,
                          object_mask=None, crop_box=None):
        ret_images = []
        ret_masks = []

        for index, img in enumerate(image):
            orig_image = tensor2pil(torch.unsqueeze(img, 0)).convert('RGB')
            width, height = orig_image.size
            black_image = Image.new('L', (width, height), color="black")

            if crop_box is not None:
                w = crop_box[2] - crop_box[0]
                h = crop_box[3] - crop_box[1]
                x1 = crop_box[0] - int(obj_ext_left * w * 0.01)
                y1 = crop_box[1] - int(obj_ext_top * h * 0.01)
                x2 = crop_box[2] + int(obj_ext_right * w * 0.01)
                y2 = crop_box[3] + int(obj_ext_bottom * h * 0.01)
                bbox = [(x1, y1, x2, y2)]
            elif object_mask is not None:
                if object_mask.dim() == 2: object_mask = torch.unsqueeze(object_mask, 0)
                mask = object_mask[index] if index < len(object_mask) else object_mask[-1]
                mask = tensor2pil(mask)
                bluredmask = gaussian_blur(mask, 20).convert('L')
                x = -10
                y = -10
                w = 4
                h = 4
                if detect == "min_bounding_rect":
                    (x, y, w, h) = min_bounding_rect(bluredmask)
                elif detect == "max_inscribed_rect":
                    (x, y, w, h) = max_inscribed_rect(bluredmask)
                else:
                    (x, y, w, h) = mask_area(mask)

                x1 = x - int(obj_ext_left * w * 0.01)
                y1 = y - int(obj_ext_top * h * 0.01)
                x2 = x + w + int(obj_ext_right * w * 0.01)
                y2 = y + h + int(obj_ext_bottom * h * 0.01)
                bbox = [(x1, y1, x2, y2)]
            else:
                bbox = [(int(left * width * 0.01),
                         int(top * height * 0.01),
                         width - int(right * width * 0.01),
                         height - int(bottom * height * 0.01))
                        ]
            rect_mask = draw_rounded_rectangle(black_image, rounded_rect_radius, bbox, anti_aliasing, "white")
            ret_image = RGB2RGBA(orig_image, rect_mask)
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(rect_mask))

        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: RoundedRectangle": LS_RoundedRectangle
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: RoundedRectangle": "LayerUtility: RoundedRectangle"
}