import torch

from .imagefunc import *
from .segment_anything_func import *

NODE_NAME = 'SegmentAnythingUltra'

class SegmentAnythingUltra:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "image": ("IMAGE",),
                "sam_model": (list_sam_model(), ),
                "grounding_dino_model": (list_groundingdino_model(),),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "detail_range": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.98, "step": 0.01}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "prompt": ("STRING", {}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "segment_anything_ultra"
    CATEGORY = '😺dzNodes/LayerMask'
  
    def segment_anything_ultra(self, image, sam_model, grounding_dino_model, threshold,
                               detail_range, black_point, white_point, process_detail,
                               prompt, ):

        sam_model = load_sam_model(sam_model)
        dino_model = load_groundingdino_model(grounding_dino_model)
        ret_images = []
        ret_masks = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            item = tensor2pil(i).convert('RGBA')
            boxes = groundingdino_predict(dino_model, item, prompt, threshold)
            if boxes.shape[0] == 0:
                break
            (_, _mask) = sam_segment(sam_model, item, boxes)
            _mask = _mask[0]
            if process_detail:
                try:
                    _mask = tensor2pil(mask_edge_detail(i, mask2image(_mask), detail_range, black_point, white_point))
                except:
                    _mask = mask2image(_mask)
            else:
                _mask = mask2image(_mask)
            _image = RGB2RGBA(tensor2pil(i).convert('RGB'), _mask.convert('L'))

            ret_images.append(pil2tensor(_image))
            ret_masks.append(image2mask(_mask))
        if len(ret_masks) == 0:
            _, height, width, _ = image.size()
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)

        log(f"{NODE_NAME} Processed {len(ret_masks)} image(s).")
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: SegmentAnythingUltra": SegmentAnythingUltra,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: SegmentAnythingUltra": "LayerMask: SegmentAnythingUltra",
}
