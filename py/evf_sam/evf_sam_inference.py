import os
import sys
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from .model.segment_anything.utils.transforms import ResizeLongestSide

def sam_preprocess(
    x: np.ndarray,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
    model_type="ori") -> torch.Tensor:
    '''
    preprocess of Segment Anything Model, including scaling, normalization and padding.  
    preprocess differs between SAM and Effi-SAM, where Effi-SAM use no padding.
    input: ndarray
    output: torch.Tensor
    '''
    assert img_size==1024, \
        "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
    x = ResizeLongestSide(img_size).apply_image(x)
    resize_shape = x.shape[:2]
    x = torch.from_numpy(x).permute(2,0,1).contiguous()

    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    if model_type=="effi" or model_type=="sam2":
        x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear").squeeze(0)
    else:
        # Pad
        h, w = x.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
    return x, resize_shape

def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    '''
    preprocess for BEIT-3 model.
    input: ndarray
    output: torch.Tensor
    '''
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)

def init_models(model_path:str, model_type:str, precision:str, load_in_bit:int=16):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        use_fast=False,
    )

    torch_dtype = torch.float32
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}

    if load_in_bit==4:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            }
        )
    elif load_in_bit==8:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    if model_type=="ori":
        from model.evf_sam import EvfSamModel
        model = EvfSamModel.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    elif model_type=="effi":
        from model.evf_effisam import EvfEffiSamModel
        model = EvfEffiSamModel.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    elif model_type=="sam2":
        from model.evf_sam2 import EvfSam2Model
        model = EvfSam2Model.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )

    if load_in_bit > 8 and torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    return tokenizer, model

def evf_sam_main(model_path:str, model_type:str, precision:str, load_in_bit:int, image:Image, prompt:str, ):

    image_size = 224
    # initialize model and tokenizer
    tokenizer, model = init_models(model_path, model_type, precision, load_in_bit)

    # preprocess
    image_np = np.asarray(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]
    image_beit = beit3_preprocess(image_np, image_size).to(dtype=model.dtype, device=model.device)
    image_sam, resize_shape = sam_preprocess(image_np, model_type=model_type)
    image_sam = image_sam.to(dtype=model.dtype, device=model.device)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)

    # infer
    pred_mask = model.inference(
        image_sam.unsqueeze(0),
        image_beit.unsqueeze(0),
        input_ids,
        resize_list=[resize_shape],
        original_size_list=original_size_list,
    )

    pred_mask = pred_mask.detach().cpu().numpy()[0]
    pred_mask = (pred_mask > 0).astype(np.uint8) * 255
    out_put_image = Image.fromarray(pred_mask.squeeze(), mode="L")

    return out_put_image