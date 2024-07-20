from transparent_background import Remover
from .imagefunc import *

NODE_NAME = 'TransparentBackgroundUltra'

mode_dict = {"ckpt_base.pth": "base", "ckpt_base_nightly.pth": "base-nightly", "ckpt_fast.pth": "fast"}
def scan_model():
    model_file_list = glob.glob(os.path.join(folder_paths.models_dir, "transparent-background") + '/*.pth')
    model_dict = {}
    for i in range(len(model_file_list)):
        _, __filename = os.path.split(model_file_list[i])
        model_dict[__filename] = model_file_list[i]
    return model_dict

class TransparentBackgroundUltra:

    @classmethod
    def INPUT_TYPES(cls):
        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda','cpu']

        return {
            "required": {
                "image": ("IMAGE",),
                "model": (list(scan_model().keys()),),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "transparent_background_ultra"
    CATEGORY = '😺dzNodes/LayerMask'
  
    def transparent_background_ultra(self, image, model, detail_method, detail_erode, detail_dilate,
                       black_point, white_point, process_detail, device, max_megapixels):
        ret_images = []
        ret_masks = []
        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False
        model_dict = scan_model()
        remover = Remover(mode=mode_dict[model], jit=False, device=device, ckpt=model_dict[model])
        for i in image:
            i = torch.unsqueeze(i, 0)
            orig_image = tensor2pil(i).convert('RGB')
            ret_image = remover.process(orig_image, type='rgba')
            _mask = ret_image.split()[3]
            _mask = adjust_levels(_mask, 64, 192)

            if process_detail:
                detail_range = detail_erode + detail_dilate
                _mask = pil2tensor(_mask)
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
                ret_image = RGB2RGBA(orig_image, _mask.convert('L'))

            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')

        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: TransparentBackgroundUltra": TransparentBackgroundUltra,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: TransparentBackgroundUltra": "LayerMask: Transparent Background Ultra",
}
