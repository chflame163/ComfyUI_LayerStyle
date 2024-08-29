'''
推理部分代码来自https://github.com/hustvl/EVF-SAM
'''
import sys

from .imagefunc import *
sys.path.append(os.path.join(os.path.dirname(__file__), 'evf_sam'))
from evf_sam.evf_sam_inference import evf_sam_main
class EVF_SAM_Ultra:

    def __init__(self):
        self.NODE_NAME = 'EVF_SAM Ultra'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        model_list = ["evf-sam2","evf-sam"]
        precision_list = ["fp16", "bf16", "fp32"]
        load_in_bit_list = ["full", "8", "4"]
        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda', 'cpu']
        return {"required":
            {
                "image": ("IMAGE",),
                "model": (model_list,),
                "precision": (precision_list,),
                "load_in_bit": (load_in_bit_list,),
                "prompt": ("STRING", {"default": "subject"}),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 4, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "evf_sam_ultra"
    CATEGORY = '😺dzNodes/LayerMask'

    def evf_sam_ultra(self, image, model, precision, load_in_bit, prompt,
                      detail_method, detail_erode, detail_dilate, black_point, white_point,
                      process_detail, device, max_megapixels,
                      ):

        ret_images = []
        ret_masks = []

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        if model == 'evf-sam2':
            model_type = 'sam2'
        elif model == 'evf-sam':
            model_type = 'ori'
        else:
            model_type = 'effi'

        if load_in_bit == 'full':
            load_in_bit = 16
        else:
            load_in_bit = int(load_in_bit)

        model_path = ""
        model_folder_name = 'EVF-SAM'
        try:
            model_path = os.path.join(
                os.path.normpath(folder_paths.folder_names_and_paths[model_folder_name][0][0]), model)
        except:
            pass
        if not os.path.exists(model_path):
            model_path = os.path.join(folder_paths.models_dir, model_folder_name, model)

        for i in image:
            i = torch.unsqueeze(i, 0)
            orig_image = tensor2pil(i).convert('RGB')
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))

            mask_image = evf_sam_main(model_path, model_type, precision, load_in_bit, orig_image, prompt)
            _mask = pil2tensor(mask_image)

            detail_range = detail_erode + detail_dilate
            if process_detail:
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device,
                                              max_megapixels=max_megapixels)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
            else:
                _mask = mask2image(_mask)

            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: EVFSAMUltra": EVF_SAM_Ultra
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: EVFSAMUltra": "LayerMask: EVF-SAM Ultra"
}

