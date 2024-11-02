from .imagefunc import *
from .segment_anything_func import *

NODE_NAME = 'SegmentAnythingUltra V2'




class SegmentAnythingUltraV2:
    def __init__(self):
        self.SAM_MODEL = None
        self.DINO_MODEL = None
        self.previous_sam_model = ""
        self.previous_dino_model = ""
        pass

    @classmethod
    def INPUT_TYPES(cls):

        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda','cpu']
        return {
            "required": {
                "image": ("IMAGE",),
                "sam_model": (list_sam_model(), ),
                "grounding_dino_model": (list_groundingdino_model(),),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "prompt": ("STRING", {"default": "subject"}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
                "cache_model": ("BOOLEAN", {"default": False}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "segment_anything_ultra_v2"
    CATEGORY = '😺dzNodes/LayerMask'

    def segment_anything_ultra_v2(self, image, sam_model, grounding_dino_model, threshold,
                                  detail_method, detail_erode, detail_dilate,
                                  black_point, white_point, process_detail, prompt,
                                  device, max_megapixels, cache_model
                                  ):

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        if self.previous_sam_model != sam_model or self.SAM_MODEL is None:
            self.SAM_MODEL = load_sam_model(sam_model)
            self.previous_sam_model = sam_model
        if self.previous_dino_model != grounding_dino_model or self.DINO_MODEL is None:
            self.DINO_MODEL = load_groundingdino_model(grounding_dino_model)
            self.previous_dino_model = grounding_dino_model

        # SAM_MODEL = load_sam_model(sam_model)
        # DINO_MODEL = load_groundingdino_model(grounding_dino_model)
        ret_images = []
        ret_masks = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            _image = tensor2pil(i).convert('RGBA')
            boxes = groundingdino_predict(self.DINO_MODEL, _image, prompt, threshold)
            if boxes.shape[0] == 0:
                break
            (_, _mask) = sam_segment(self.SAM_MODEL, _image, boxes)
            _mask = _mask[0]
            detail_range = detail_erode + detail_dilate
            if process_detail:
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(_image, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
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
    "LayerMask: SegmentAnythingUltra V2": SegmentAnythingUltraV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: SegmentAnythingUltra V2": "LayerMask: SegmentAnythingUltra V2",
}
