# layerstyle advance
import folder_paths
from .imagefunc import *

any = AnyType("*")

class LS_DrawRoundedRectangle:

    def __init__(self):
        self.NODE_NAME = 'Draw Rounded Rectangle'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rounded_rect_radius": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "anti_aliasing": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 8, "max": 1e8, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 1e8, "step": 1}),
            },
            "optional": {
                "size_as": (any, {}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'draw_rounded_rectangle'
    CATEGORY = '😺dzNodes/LayerMask'

    def draw_rounded_rectangle(self, rounded_rect_radius, anti_aliasing, width, height, size_as=None,):

        if size_as is None:
            target_width, target_height = width, height
        else:
            if size_as.shape[0] > 0:
                _asimage = tensor2pil(size_as[0])
            else:
                _asimage = tensor2pil(size_as)
            target_width, target_height = _asimage.size

        ret_masks = []
        mask = Image.new("L", (target_width, target_height), color='black')
        bboxes = [(0, 0, target_width, target_height)]
        mask = draw_rounded_rectangle(mask, rounded_rect_radius, bboxes, anti_aliasing)
        ret_masks.append(pil2tensor(mask))

        log(f"{self.NODE_NAME} Processed {len(ret_masks)} mask(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerMask: DrawRoundedRectangle": LS_DrawRoundedRectangle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: DrawRoundedRectangle": "LayerMask: DrawRoundedRectangle",
}