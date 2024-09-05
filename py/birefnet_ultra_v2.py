import os
import sys
import torch
from torchvision import transforms
import tqdm
from .imagefunc import *
from comfy.utils import ProgressBar
sys.path.append(os.path.join(os.path.dirname(__file__), 'BiRefNet'))
from .BiRefNet.models.birefnet import BiRefNet

def get_models():
    model_path = os.path.join(folder_paths.models_dir, 'BiRefNet', 'pth')
    model_ext = [".pth"]
    model_dict = get_files(model_path, model_ext)
    return model_dict

class LS_LoadBiRefNetModel:

    def __init__(self):
        self.birefnet = None
        self.state_dict = None


    @classmethod
    def INPUT_TYPES(s):
        tmp_list = list(get_models().keys())
        model_list = []
        if 'BiRefNet-general-epoch_244.pth' in tmp_list:
            model_list.append('BiRefNet-general-epoch_244.pth')
            tmp_list.remove('BiRefNet-general-epoch_244.pth')
        model_list.extend(tmp_list)

        return {
            "required": {
                "model": (model_list,),
            },
        }

    RETURN_TYPES = ("BIREFNET_MODEL",)
    RETURN_NAMES = ("birefnet_model",)
    FUNCTION = "load_birefnet_model"
    CATEGORY = '😺dzNodes/LayerMask'

    def load_birefnet_model(self, model):
        from .BiRefNet.utils import check_state_dict
        model_dict = get_models()
        self.birefnet = BiRefNet(bb_pretrained=False)
        self.state_dict = torch.load(model_dict[model], map_location='cpu')
        self.state_dict = check_state_dict(self.state_dict)
        self.birefnet.load_state_dict(self.state_dict)
        return (self.birefnet,)


class LS_BiRefNetUltraV2:

    def __init__(self):
        self.NODE_NAME = 'BiRefNetUltraV2'

    @classmethod
    def INPUT_TYPES(cls):

        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda','cpu']
        return {
            "required": {
                "image": ("IMAGE",),
                "birefnet_model":("BIREFNET_MODEL",),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 4, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 2, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": False}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "birefnet_ultra_v2"
    CATEGORY = '😺dzNodes/LayerMask'
  
    def birefnet_ultra_v2(self, image, birefnet_model, detail_method, detail_erode, detail_dilate,
                       black_point, white_point, process_detail, device, max_megapixels):
        ret_images = []
        ret_masks = []
        inference_image_size = (1024, 1024)
        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        torch.set_float32_matmul_precision(['high', 'highest'][0])
        birefnet_model.to(device)
        birefnet_model.eval()

        comfy_pbar = ProgressBar(len(image))
        tqdm_pbar = tqdm(total=len(image), desc="Processing BiRefNet")
        for i in image:
            i = torch.unsqueeze(i, 0)
            orig_image = tensor2pil(i).convert('RGB')

            transform_image = transforms.Compose([
                transforms.Resize(inference_image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            inference_image = transform_image(orig_image).unsqueeze(0).to(device)

            # Prediction
            with torch.no_grad():
                preds = birefnet_model(inference_image)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            _mask = pred_pil.resize(inference_image_size)

            resize_sampler = Image.BILINEAR
            _mask = _mask.resize(orig_image.size, resize_sampler)
            brightness_image = ImageEnhance.Brightness(_mask)
            _mask = brightness_image.enhance(factor=1.01)
            _mask = image2mask(_mask)

            detail_range = detail_erode + detail_dilate

            if process_detail:
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
            else:
                _mask = tensor2pil(_mask)

            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

            comfy_pbar.update(1)
            tqdm_pbar.update(1)

        log(f"{self.NODE_NAME} Processed {len(ret_masks)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: BiRefNetUltraV2": LS_BiRefNetUltraV2,
    "LayerMask: LoadBiRefNetModel": LS_LoadBiRefNetModel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: BiRefNetUltraV2": "LayerMask: BiRefNet Ultra V2",
    "LayerMask: LoadBiRefNetModel": "LayerMask: Load BiRefNet Model"
}
