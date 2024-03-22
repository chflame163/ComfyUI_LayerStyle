import torch

from .imagefunc import *

import torch.nn as nn
from torchvision import transforms
from .BiRefNet.baseline import BiRefNet
# from .BiRefNet import config
from .BiRefNet.config import Config

NODE_NAME = 'BiRefNetUltra'


config = Config()

class BiRefNet_img_processor:
    def __init__(self, config):
        self.config = config
        self.data_size = (config.size, config.size)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, _image: np.array):
        _image_rs = cv2.resize(_image, (self.config.size, self.config.size), interpolation=cv2.INTER_LINEAR)
        _image_rs = Image.fromarray(np.uint8(_image_rs*255)).convert('RGB')
        image = self.transform_image(_image_rs)
        return image

class BiRefNetUltra:
    def __init__(self):
        self.ready = False

    def load(self, weight_path, device):
        # load model
        self.model = BiRefNet()
        state_dict = torch.load(weight_path, map_location='cpu')
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()
        # load processor
        self.processor = BiRefNet_img_processor(config)
        self.ready = True


    @classmethod
    def INPUT_TYPES(cls):

        method_list = ['VITMatte', 'PyMatting', 'GuidedFilter']


        return {
            "required": {
                "image": ("IMAGE",),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01}),
                "process_detail": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "birefnet_ultra"
    CATEGORY = '😺dzNodes/LayerMask'
  
    def birefnet_ultra(self, image, detail_method, detail_erode, detail_dilate,
                       black_point, white_point, process_detail):
        ret_images = []
        ret_masks = []

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        if not self.ready:
            model_folder_name = 'BiRefNet'
            model_name = 'BiRefNet-ep480.pth'
            model_file_path = ""
            try:
                model_file_path = os.path.join(
                    os.path.normpath(folder_paths.folder_names_and_paths[model_folder_name][0][0]), model_name)
            except:
                pass
            if not os.path.exists(model_file_path):
                model_file_path = os.path.join(folder_paths.models_dir, model_folder_name, model_name)
            self.load(model_file_path, device=device)

        for i in image:
            i = torch.unsqueeze(i, 0)
            orig_image = tensor2pil(i).convert('RGB')
            np_image = i.squeeze().numpy()
            img = self.processor(np_image)
            inputs = img[None, ...].to(device)
            with torch.no_grad():
                scaled_preds = self.model(inputs)[-1].sigmoid()
            _mask = nn.functional.interpolate(
                scaled_preds[0].unsqueeze(0),
                size=np_image.shape[:2],
                mode='bilinear',
                align_corners=True
            )[0]

            detail_range = detail_erode + detail_dilate
            if process_detail:
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    brightness_image = ImageEnhance.Brightness(tensor2pil(_mask))
                    _mask = brightness_image.enhance(factor=1.01)
                    _mask = pil2tensor(_mask)
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(orig_image, _trimap)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
            else:
                _mask = tensor2pil(_mask)

            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{NODE_NAME} Processed {len(ret_masks)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)
        # return (None, torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: BiRefNetUltra": BiRefNetUltra,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: BiRefNetUltra": "LayerMask: BiRefNetUltra",
}
