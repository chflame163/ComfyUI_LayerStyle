from .imagefunc import *

import torch.nn as nn
from torchvision import transforms
from .BiRefNet_legacy.baseline import BiRefNet
from .BiRefNet_legacy.config import Config

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

class BiRefNetRemoveBackground:
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
        self.processor = BiRefNet_img_processor(Config())
        self.ready = True

  
    def generate_mask(self, image:Image) -> Image:

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

        i = pil2tensor(image)
        orig_image = image.convert('RGB')
        np_image = i.squeeze().numpy()
        img = self.processor(np_image)
        inputs = img[None, ...].to(device)
        with torch.no_grad():
            scaled_preds = self.model(inputs)[-1].sigmoid()
        _mask = nn.functional.interpolate(scaled_preds[0].unsqueeze(0),
                                          size=np_image.shape[:2],
                                          mode='bilinear',
                                          align_corners=True
                                          )[0]

        brightness_image = ImageEnhance.Brightness(tensor2pil(_mask))

        return brightness_image.enhance(factor=1.01)
