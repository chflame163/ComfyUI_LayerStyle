import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, image2mask, mask2image, expand_mask, mask_fix, gaussian_blur, pixel_spread
from  .imagefunc import guided_filter_alpha, histogram_remap, mask_edge_detail ,RGB2RGBA, generate_VITMatte, generate_VITMatte_trimap



class MaskEdgeUltraDetailV3:
    def __init__(self):
        self.NODE_NAME = 'MaskEdgeUltraDetail V3'

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
                "mask_edge_erode": ("INT", {"default": 6, "min": 1, "max": 512, "step": 1}),
                "mask_edge_dilate": ("INT", {"default": 4, "min": 1, "max": 512, "step": 1}),
                "transparent_trimap_erode": ("INT", {"default": 72, "min": 1, "max": 1024, "step": 1}),
                "transparent_trimap_dilate": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
                "trimap_blur": ("INT", {"default": 4, "min": 1, "max": 512, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "spread_mask_grow": ("INT", {"default": 0, "min": -999, "max": 999, "step": 1}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 3.0, "min": 1, "max": 999, "step": 0.1}),
            },
            "optional": {
                "transparent_trimap": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "mask_edge_ultra_detail_v3"
    CATEGORY = '😺dzNodes/LayerMask'
  
    def mask_edge_ultra_detail_v3(self, image, mask, method, mask_grow, fix_gap, fix_threshold,
                                  mask_edge_erode, mask_edge_dilate, transparent_trimap_erode, transparent_trimap_dilate, trimap_blur,
                                  black_point, white_point, spread_mask_grow,
                                  device, max_megapixels,
                                  transparent_trimap=None):
        ret_images = []
        ret_masks = []
        l_images = []
        l_masks = []
        trimap_masks = []

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
        if transparent_trimap is not None:
            if transparent_trimap.dim() == 2:
                transparent_trimap = torch.unsqueeze(transparent_trimap, 0)
            for t in transparent_trimap:
                trimap_masks.append(torch.unsqueeze(t, 0))

        if len(l_images) != len(l_masks) or tensor2pil(l_images[0]).size != tensor2pil(l_masks[0]).size:
            log(f"Error: {self.NODE_NAME} skipped, because mask does'nt match image.", message_type='error')
            return (image, mask,)
        detail_range = mask_edge_erode + mask_edge_dilate
        trimap_detail_range = transparent_trimap_erode + transparent_trimap_dilate
        for i in range(len(l_images)):
            _image = l_images[i]
            orig_image = tensor2pil(_image).convert('RGB')
            _image = pil2tensor(orig_image)
            _mask = l_masks[i]

            if mask_grow != 0:
                _mask = expand_mask(_mask, mask_grow, mask_grow//2)
            if fix_gap:
                _mask = mask_fix(_mask, 1, fix_gap, fix_threshold, fix_threshold)
            log(f"{self.NODE_NAME} Processing...")
            if method == 'GuidedFilter':
                processed_mask = guided_filter_alpha(_image, _mask, detail_range//6)
                processed_mask = tensor2pil(histogram_remap(processed_mask, black_point, white_point))
                if transparent_trimap is not None:
                    processed_trimap = guided_filter_alpha(_image, _mask, trimap_detail_range//6)
                    processed_trimap = tensor2pil(histogram_remap(processed_trimap, black_point, white_point))
            elif method == 'PyMatting':
                processed_mask = tensor2pil(mask_edge_detail(_image, _mask, detail_range//8, black_point, white_point))
                if transparent_trimap is not None:
                    processed_trimap = tensor2pil(mask_edge_detail(_image, _mask, trimap_detail_range//8, black_point, white_point))
            else:
                _trimap = generate_VITMatte_trimap(_mask, mask_edge_erode, mask_edge_dilate)
                processed_mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                processed_mask = tensor2pil(histogram_remap(pil2tensor(processed_mask), black_point, white_point))
                if transparent_trimap is not None:
                    _trimap = generate_VITMatte_trimap(_mask, transparent_trimap_erode, transparent_trimap_dilate)
                    processed_trimap = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                    processed_trimap = tensor2pil(histogram_remap(pil2tensor(processed_trimap), black_point, white_point))

            if transparent_trimap is not None:
                _trimap_mask = tensor2pil(trimap_masks[i]).convert('L')
                if trimap_blur > 0:
                    _trimap_mask = gaussian_blur(_trimap_mask, trimap_blur)
                processed_mask.paste(processed_trimap, mask=_trimap_mask)

            if spread_mask_grow != 0:
                spread_mask = expand_mask(image2mask(processed_mask), spread_mask_grow, 0)  # 扩张，模糊
                spread_mask = mask2image(spread_mask)
            else:
                spread_mask = processed_mask

            ret_image = pixel_spread(orig_image.convert('RGB'), spread_mask.convert('RGB'))
            ret_image = RGB2RGBA(ret_image, processed_mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(processed_mask))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskEdgeUltraDetail V3": MaskEdgeUltraDetailV3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskEdgeUltraDetail V3": "LayerMask: MaskEdgeUltraDetail V3",
}
