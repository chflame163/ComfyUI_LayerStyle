import torch
import random
from nodes import SaveImage
import folder_paths


class MaskPreview(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz1234567890") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {"mask": ("MASK",), },
        }

    FUNCTION = "mask_preview"
    CATEGORY = '😺dzNodes/LayerMask'
    OUTPUT_NODE = True

    def mask_preview(self, mask):
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return self.save_images(preview, "MaskPreview")

NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskPreview": MaskPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskPreview": "LayerMask: MaskPreview"
}