from .imagefunc import *

NODE_NAME = 'MaskEdgeUltraDetail V2'

class MaskEdgeUltraDetailV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda','cpu']
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "method": (method_list,),
                "mask_grow": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "fix_gap": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "fix_threshold": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 0.99, "step": 0.01}),
                "edge_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "edte_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "mask_edge_ultra_detail_v2"
    CATEGORY = '😺dzNodes/LayerMask'
  
    def mask_edge_ultra_detail_v2(self, image, mask, method, mask_grow, fix_gap, fix_threshold,
                               edge_erode, edte_dilate, black_point, white_point, device, max_megapixels,):
        ret_images = []
        ret_masks = []
        l_images = []
        l_masks = []

        if method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
        for m in mask:
            l_masks.append(torch.unsqueeze(m, 0))
        if len(l_images) != len(l_masks) or tensor2pil(l_images[0]).size != tensor2pil(l_masks[0]).size:
            log(f"Error: {NODE_NAME} skipped, because mask does'nt match image.", message_type='error')
            return (image, mask,)
        detail_range = edge_erode + edte_dilate
        for i in range(len(l_images)):
            _image = l_images[i]
            orig_image = tensor2pil(_image).convert('RGB')
            _image = pil2tensor(orig_image)
            _mask = l_masks[i]
            if mask_grow != 0:
                _mask = expand_mask(_mask, mask_grow, mask_grow//2)
            if fix_gap:
                _mask = mask_fix(_mask, 1, fix_gap, fix_threshold, fix_threshold)
            log(f"{NODE_NAME} Processing...")
            if method == 'GuidedFilter':
                _mask = guided_filter_alpha(_image, _mask, detail_range//6)
                _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
            elif method == 'PyMatting':
                _mask = tensor2pil(mask_edge_detail(_image, _mask, detail_range//8, black_point, white_point))
            else:
                _trimap = generate_VITMatte_trimap(_mask, edge_erode, edte_dilate)
                _mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))

            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskEdgeUltraDetail V2": MaskEdgeUltraDetailV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskEdgeUltraDetail V2": "LayerMask: MaskEdgeUltraDetail V2",
}
