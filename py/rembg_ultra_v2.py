from .imagefunc import *

NODE_NAME = 'RmBgUltra V2'

class RmBgUltraV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        method_list = ['VITMatte', 'PyMatting', 'GuidedFilter']

        return {
            "required": {
                "image": ("IMAGE",),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 50, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 20, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01}),
                "process_detail": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "rmbg_ultra_v2"
    CATEGORY = '😺dzNodes/LayerMask'
  
    def rmbg_ultra_v2(self, image, detail_method, detail_erode, detail_dilate,
                       black_point, white_point, process_detail):
        ret_images = []
        ret_masks = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            orig_image = tensor2pil(i).convert('RGB')
            _mask = RMBG(orig_image)
            _mask = pil2tensor(_mask)

            detail_range = detail_erode + detail_dilate
            if process_detail:
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(orig_image, _trimap)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
            else:
                _mask = mask2image(_mask)

            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: RmBgUltra V2": RmBgUltraV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: RmBgUltra V2": "LayerMask: RmBgUltra V2",
}
