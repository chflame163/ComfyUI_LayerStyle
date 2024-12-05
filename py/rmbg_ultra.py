import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, image2mask, mask2image, RMBG, RGB2RGBA, mask_edge_detail


class RemBgUltra:
    def __init__(self):
        self.NODE_NAME = 'RemBgUltra'

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "image": ("IMAGE",),
                "detail_range": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01}),
                "process_detail": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "rembg_ultra"
    CATEGORY = '😺dzNodes/LayerMask'
  
    def rembg_ultra(self, image, detail_range, black_point, white_point, process_detail):
        ret_images = []
        ret_masks = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            orig_image = tensor2pil(i).convert('RGB')
            _mask = RMBG(orig_image)
            if process_detail:
                _mask = tensor2pil(mask_edge_detail(i, pil2tensor(_mask), detail_range, black_point, white_point))
            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: RemBgUltra": RemBgUltra,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: RemBgUltra": "LayerMask: RemBgUltra",
}
