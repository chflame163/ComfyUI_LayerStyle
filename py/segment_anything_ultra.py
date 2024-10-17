from .imagefunc import *
from .segment_anything_func import *

NODE_NAME = 'SegmentAnythingUltra'

class SegmentAnythingUltra:
    def __init__(self):
        self.SAM_MODEL = None
        self.DINO_MODEL = None
        self.previous_sam_model = ""
        self.previous_dino_model = ""

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "image": ("IMAGE",),
                "sam_model": (list_sam_model(), ),
                "grounding_dino_model": (list_groundingdino_model(),),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "detail_range": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.98, "step": 0.01}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "prompt": ("STRING", {"default": "subject"}),
                "cache_model": ("BOOLEAN", {"default": False}),
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
                               prompt, cache_model):

        if self.previous_sam_model != sam_model or self.SAM_MODEL is None:
            self.SAM_MODEL = load_sam_model(sam_model)
            self.previous_sam_model = sam_model
        if self.previous_dino_model != grounding_dino_model or self.DINO_MODEL is None:
            self.DINO_MODEL = load_groundingdino_model(grounding_dino_model)
            self.previous_dino_model = grounding_dino_model


        ret_images = []
        ret_masks = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            item = tensor2pil(i).convert('RGBA')
            boxes = groundingdino_predict(self.DINO_MODEL, item, prompt, threshold)
            if boxes.shape[0] == 0:
                break
            (_, _mask) = sam_segment(self.SAM_MODEL, item, boxes)
            _mask = _mask[0]
            if process_detail:
                _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range, black_point, white_point))
            else:
                _mask = mask2image(_mask)
            _image = RGB2RGBA(tensor2pil(i).convert('RGB'), _mask.convert('L'))

            ret_images.append(pil2tensor(_image))
            ret_masks.append(image2mask(_mask))
        if len(ret_masks) == 0:
            _, height, width, _ = image.size()
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)

        if not cache_model:
            self.SAM_MODEL = None
            self.DINO_MODEL = None
            self.previous_sam_model = ""
            self.previous_dino_model = ""
            clear_memory()

        log(f"{NODE_NAME} Processed {len(ret_masks)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: SegmentAnythingUltra": SegmentAnythingUltra,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: SegmentAnythingUltra": "LayerMask: SegmentAnythingUltra",
}
