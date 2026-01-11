class AdjustBBoxToImageRatio:

    def __init__(self):
        self.NODE_NAME = 'AdjustBBoxToImageRatio'

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "crop_box": ("BOX",),
                "img_width": ("INT", {"default": 1920}),
                "img_height": ("INT", {"default": 1080}),
            }
        }

    RETURN_TYPES = ("BOX",)
    RETURN_NAMES = ("crop_box",)
    FUNCTION = "adjust_bbox"
    CATEGORY = "😺dzNodes/LayerUtility"

    def adjust_bbox(self, crop_box, img_width, img_height):
        x1, y1, x2, y2 = crop_box
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2

        target_ratio = img_width / img_height
        box_ratio = w / h

        if box_ratio > target_ratio:
            new_h = w / target_ratio
            new_w = w
        else:
            new_w = h * target_ratio
            new_h = h

        new_w = min(new_w, img_width)
        new_h = min(new_h, img_height)

        cx = max(new_w / 2, min(cx, img_width - new_w / 2))
        cy = max(new_h / 2, min(cy, img_height - new_h / 2))

        new_x = cx - new_w / 2
        new_y = cy - new_h / 2

        return ((int(new_x), int(new_y), int(new_x + new_w), int(new_y + new_h)),)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: AdjustBBoxToImageRatio": AdjustBBoxToImageRatio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: AdjustBBoxToImageRatio": "LayerUtility: AdjustBBoxToImageRatio",
}