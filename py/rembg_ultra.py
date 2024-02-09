
import copy
import torch, os
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pymatting import *
# from torchvision.transforms.functional import normalize
import torchvision.transforms.functional as TF
from .briarmbg import BriaRMBG
from .imagefunc import *

current_directory = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    net = BriaRMBG()
    model_path = os.path.join(os.path.dirname(current_directory), "RMBG-1.4/model.pth")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    return net

class RemBgUltra:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "image": ("IMAGE",),
                "detail_range": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1}),
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

        rmbgmodel = load_model()
        orig_image = tensor2pil(image).convert('RGB')
        w,h = orig_image.size
        im_np = np.array(orig_image.resize((1024, 1024), Image.BILINEAR))
        im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
        im_tensor = torch.unsqueeze(im_tensor,0)
        im_tensor = torch.divide(im_tensor,255.0)
        # im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
        im_tensor = TF.normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        if torch.cuda.is_available():
            im_tensor=im_tensor.cuda()
        result=rmbgmodel(im_tensor)
        result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear') ,0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result-mi)/(ma-mi)
        im_array = (result*255).cpu().data.numpy().astype(np.uint8)
        _mask = Image.fromarray(np.squeeze(im_array)).convert('L')
        if process_detail:
            # ultra edge process
            d = detail_range * 2 + 1
            i_dup = copy.deepcopy(image.cpu().numpy().astype(np.float64))
            a_dup = copy.deepcopy(pil2tensor(_mask.convert('RGB')).cpu().numpy().astype(np.float64))
            for index, img in enumerate(i_dup):
                trimap = a_dup[index][:,:,0] # convert to single channel
                if detail_range > 0:
                    trimap = cv2.GaussianBlur(trimap, (d, d), 0)
                trimap = fix_trimap(trimap, black_point, white_point)
                alpha = estimate_alpha_cf(img, trimap, laplacian_kwargs={"epsilon": 1e-6},
                                          cg_kwargs={"maxiter": 500})
                a_dup[index] = np.stack([alpha, alpha, alpha], axis=-1)  # convert back to rgb
            _mask = tensor2pil(torch.from_numpy(a_dup.astype(np.float32))) # alpha
        ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
        return (pil2tensor(ret_image), image2mask(_mask),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: RemBgUltra": RemBgUltra,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: RemBgUltra": "LayerMask: RemBgUltra",
}
