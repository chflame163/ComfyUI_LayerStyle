'''
原始代码来自 https://github.com/StartHua/Comfyui_segformer_b2_clothes
'''
import torch
import os
import numpy as np
from PIL import Image, ImageEnhance
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn
import folder_paths
from .imagefunc import log, tensor2pil, pil2tensor, mask2image, image2mask, RGB2RGBA
from .imagefunc import guided_filter_alpha, mask_edge_detail, histogram_remap, generate_VITMatte, generate_VITMatte_trimap


class SegformerPipeline:
    def __init__(self):
        self.model_name = ''
        self.segment_label = []

SegPipeline = SegformerPipeline()


# 切割服装
def get_segmentation_from_model(tensor_image, segformer_model):

    processor = segformer_model["processor"]
    model = segformer_model["model"]

    cloth = tensor2pil(tensor_image)

    # 预处理和预测
    inputs = processor(images=cloth, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits, size=cloth.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    return pred_seg,cloth


# 切割服装
def get_segmentation(tensor_image, model_name='segformer_b2_clothes'):
    cloth = tensor2pil(tensor_image)
    model_folder_path = os.path.join(folder_paths.models_dir, model_name)
    try:
        model_folder_path = os.path.normpath(folder_paths.folder_names_and_paths[model_name][0][0])
    except:
        pass

    processor = SegformerImageProcessor.from_pretrained(model_folder_path)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_folder_path)
    # 预处理和预测
    inputs = processor(images=cloth, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits, size=cloth.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    return pred_seg,cloth


class Segformer_B2_Clothes:

    def __init__(self):
        self.NODE_NAME = 'SegformerB2ClothesUltra'


    # Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt",
    # 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
    # 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"

    @classmethod
    def INPUT_TYPES(cls):
        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda', 'cpu']
        return {"required":
            {
                "image": ("IMAGE",),
                "face": ("BOOLEAN", {"default": False}),
                "hair": ("BOOLEAN", {"default": False}),
                "hat": ("BOOLEAN", {"default": False}),
                "sunglass": ("BOOLEAN", {"default": False}),
                "left_arm": ("BOOLEAN", {"default": False}),
                "right_arm": ("BOOLEAN", {"default": False}),
                "left_leg": ("BOOLEAN", {"default": False}),
                "right_leg": ("BOOLEAN", {"default": False}),
                "upper_clothes": ("BOOLEAN", {"default": False}),
                "skirt": ("BOOLEAN", {"default": False}),
                "pants": ("BOOLEAN", {"default": False}),
                "dress": ("BOOLEAN", {"default": False}),
                "belt": ("BOOLEAN", {"default": False}),
                "shoe": ("BOOLEAN", {"default": False}),
                "bag": ("BOOLEAN", {"default": False}),
                "scarf": ("BOOLEAN", {"default": False}),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 12, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": (
                "FLOAT", {"default": 0.15, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": (
                "FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "segformer_ultra"
    CATEGORY = '😺dzNodes/LayerMask'

    def segformer_ultra(self, image,
                        face, hat, hair, sunglass, upper_clothes, skirt, pants, dress, belt, shoe,
                        left_leg, right_leg, left_arm, right_arm, bag, scarf, detail_method,
                        detail_erode, detail_dilate, black_point, white_point, process_detail, device, max_megapixels,
                        ):

        ret_images = []
        ret_masks = []

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        for i in image:
            pred_seg, cloth = get_segmentation(i)
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            orig_image = tensor2pil(i).convert('RGB')

            labels_to_keep = [0]
            if not hat:
                labels_to_keep.append(1)
            if not hair:
                labels_to_keep.append(2)
            if not sunglass:
                labels_to_keep.append(3)
            if not upper_clothes:
                labels_to_keep.append(4)
            if not skirt:
                labels_to_keep.append(5)
            if not pants:
                labels_to_keep.append(6)
            if not dress:
                labels_to_keep.append(7)
            if not belt:
                labels_to_keep.append(8)
            if not shoe:
                labels_to_keep.append(9)
                labels_to_keep.append(10)
            if not face:
                labels_to_keep.append(11)
            if not left_leg:
                labels_to_keep.append(12)
            if not right_leg:
                labels_to_keep.append(13)
            if not left_arm:
                labels_to_keep.append(14)
            if not right_arm:
                labels_to_keep.append(15)
            if not bag:
                labels_to_keep.append(16)
            if not scarf:
                labels_to_keep.append(17)

            mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)

            # 创建agnostic-mask图像
            mask_image = Image.fromarray((1 - mask) * 255)
            mask_image = mask_image.convert("L")
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

class SegformerClothesPipelineLoader:

    def __init__(self):
        self.NODE_NAME = 'SegformerClothesPipelineLoader'
        pass

    # Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes",
    # 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe",
    # 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
    #  17: "Scarf"

    @classmethod
    def INPUT_TYPES(cls):
        model_list = ['segformer_b3_clothes', 'segformer_b2_clothes']
        return {"required":
            {   "model": (model_list,),
                "face": ("BOOLEAN", {"default": False, "label_on": "enabled(脸)", "label_off": "disabled(脸)"}),
                "hair": ("BOOLEAN", {"default": False, "label_on": "enabled(头发)", "label_off": "disabled(头发)"}),
                "hat": ("BOOLEAN", {"default": False, "label_on": "enabled(帽子)", "label_off": "disabled(帽子)"}),
                "sunglass": ("BOOLEAN", {"default": False, "label_on": "enabled(墨镜)", "label_off": "disabled(墨镜)"}),
                "left_arm": ("BOOLEAN", {"default": False, "label_on": "enabled(左臂)", "label_off": "disabled(左臂)"}),
                "right_arm": ("BOOLEAN", {"default": False, "label_on": "enabled(右臂)", "label_off": "disabled(右臂)"}),
                "left_leg": ("BOOLEAN", {"default": False, "label_on": "enabled(左腿)", "label_off": "disabled(左腿)"}),
                "right_leg": ("BOOLEAN", {"default": False, "label_on": "enabled(右腿)", "label_off": "disabled(右腿)"}),
                "left_shoe": ("BOOLEAN", {"default": False, "label_on": "enabled(左鞋)", "label_off": "disabled(左鞋)"}),
                "right_shoe": ("BOOLEAN", {"default": False, "label_on": "enabled(右鞋)", "label_off": "disabled(右鞋)"}),
                "upper_clothes": ("BOOLEAN", {"default": False, "label_on": "enabled(上衣)", "label_off": "disabled(上衣)"}),
                "skirt": ("BOOLEAN", {"default": False, "label_on": "enabled(短裙)", "label_off": "disabled(短裙)"}),
                "pants": ("BOOLEAN", {"default": False, "label_on": "enabled(裤子)", "label_off": "disabled(裤子)"}),
                "dress": ("BOOLEAN", {"default": False, "label_on": "enabled(连衣裙)", "label_off": "disabled(连衣裙)"}),
                "belt": ("BOOLEAN", {"default": False, "label_on": "enabled(腰带)", "label_off": "disabled(腰带)"}),
                "bag": ("BOOLEAN", {"default": False, "label_on": "enabled(背包)", "label_off": "disabled(背包)"}),
                "scarf": ("BOOLEAN", {"default": False, "label_on": "enabled(围巾)", "label_off": "disabled(围巾)"}),
            }
        }

    RETURN_TYPES = ("SegPipeline",)
    RETURN_NAMES = ("segformer_pipeline",)
    FUNCTION = "segformer_clothes_pipeline_loader"
    CATEGORY = '😺dzNodes/LayerMask'

    def segformer_clothes_pipeline_loader(self, model,
                        face, hat, hair, sunglass,
                        left_leg, right_leg, left_arm, right_arm, left_shoe, right_shoe,
                        upper_clothes, skirt, pants, dress, belt, bag, scarf,
                        ):

        pipeline = SegformerPipeline()
        labels_to_keep = [0]
        if not hat:
            labels_to_keep.append(1)
        if not hair:
            labels_to_keep.append(2)
        if not sunglass:
            labels_to_keep.append(3)
        if not upper_clothes:
            labels_to_keep.append(4)
        if not skirt:
            labels_to_keep.append(5)
        if not pants:
            labels_to_keep.append(6)
        if not dress:
            labels_to_keep.append(7)
        if not belt:
            labels_to_keep.append(8)
        if not left_shoe:
            labels_to_keep.append(9)
        if not right_shoe:
            labels_to_keep.append(10)
        if not face:
            labels_to_keep.append(11)
        if not left_leg:
            labels_to_keep.append(12)
        if not right_leg:
            labels_to_keep.append(13)
        if not left_arm:
            labels_to_keep.append(14)
        if not right_arm:
            labels_to_keep.append(15)
        if not bag:
            labels_to_keep.append(16)
        if not scarf:
            labels_to_keep.append(17)
        pipeline.segment_label = labels_to_keep
        pipeline.model_name = model
        return (pipeline,)

class SegformerFashionPipelineLoader:

    def __init__(self):
        self.NODE_NAME = 'SegformerFashionPipelineLoader'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        model_list = ['segformer_b3_fashion']
        return {"required":
            {   "model": (model_list,),
                "shirt": ("BOOLEAN", {"default": False, "label_on": "enabled(衬衫、罩衫)", "label_off": "disabled(衬衫、罩衫)"}),
                "top": ("BOOLEAN", {"default": False, "label_on": "enabled(上衣、t恤)", "label_off": "disabled(上衣、t恤)"}),
                "sweater": ("BOOLEAN", {"default": False, "label_on": "enabled(毛衣)", "label_off": "disabled(毛衣)"}),
                "cardigan": ("BOOLEAN", {"default": False, "label_on": "enabled(开襟毛衫)", "label_off": "disabled(开襟毛衫)"}),
                "jacket": ("BOOLEAN", {"default": False, "label_on": "enabled(夹克)", "label_off": "disabled(夹克)"}),
                "vest": ("BOOLEAN", {"default": False, "label_on": "enabled(背心)", "label_off": "disabled(背心)"}),
                "pants": ("BOOLEAN", {"default": False, "label_on": "enabled(裤子)", "label_off": "disabled(裤子)"}),
                "shorts": ("BOOLEAN", {"default": False, "label_on": "enabled(短裤)", "label_off": "disabled(短裤)"}),
                "skirt": ("BOOLEAN", {"default": False, "label_on": "enabled(裙子)", "label_off": "disabled(裙子)"}),
                "coat": ("BOOLEAN", {"default": False, "label_on": "enabled(外套)", "label_off": "disabled(外套)"}),
                "dress": ("BOOLEAN", {"default": False, "label_on": "enabled(连衣裙)", "label_off": "disabled(连衣裙)"}),
                "jumpsuit": ("BOOLEAN", {"default": False, "label_on": "enabled(连身裤)", "label_off": "disabled(连身裤)"}),
                "cape": ("BOOLEAN", {"default": False, "label_on": "enabled(斗篷)", "label_off": "disabled(斗篷)"}),
                "glasses": ("BOOLEAN", {"default": False, "label_on": "enabled(眼镜)", "label_off": "disabled(眼镜)"}),
                "hat": ("BOOLEAN", {"default": False, "label_on": "enabled(帽子)", "label_off": "disabled(帽子)"}),
                "hairaccessory": ("BOOLEAN", {"default": False, "label_on": "enabled(头带)", "label_off": "disabled(头带)"}),
                "tie": ("BOOLEAN", {"default": False, "label_on": "enabled(领带)", "label_off": "disabled(领带)"}),
                "glove": ("BOOLEAN", {"default": False, "label_on": "enabled(手套)", "label_off": "disabled(手套)"}),
                "watch": ("BOOLEAN", {"default": False, "label_on": "enabled(手表)", "label_off": "disabled(手表)"}),
                "belt": ("BOOLEAN", {"default": False, "label_on": "enabled(皮带)", "label_off": "disabled(皮带)"}),
                "legwarmer": ("BOOLEAN", {"default": False, "label_on": "enabled(腿套)", "label_off": "disabled(腿套)"}),
                "tights": ("BOOLEAN", {"default": False, "label_on": "enabled(裤袜)","label_off": "disabled(裤袜)"}),
                "sock": ("BOOLEAN", {"default": False, "label_on": "enabled(袜子)", "label_off": "disabled(袜子)"}),
                "shoe": ("BOOLEAN", {"default": False, "label_on": "enabled(鞋子)", "label_off": "disabled(鞋子)"}),
                "bagwallet": ("BOOLEAN", {"default": False, "label_on": "enabled(手包)", "label_off": "disabled(手包)"}),
                "scarf": ("BOOLEAN", {"default": False, "label_on": "enabled(围巾)", "label_off": "disabled(围巾)"}),
                "umbrella": ("BOOLEAN", {"default": False, "label_on": "enabled(雨伞)", "label_off": "disabled(雨伞)"}),
                "hood": ("BOOLEAN", {"default": False, "label_on": "enabled(兜帽)", "label_off": "disabled(兜帽)"}),
                "collar": ("BOOLEAN", {"default": False, "label_on": "enabled(衣领)", "label_off": "disabled(衣领)"}),
                "lapel": ("BOOLEAN", {"default": False, "label_on": "enabled(翻领)", "label_off": "disabled(翻领)"}),
                "epaulette": ("BOOLEAN", {"default": False, "label_on": "enabled(肩章)", "label_off": "disabled(肩章)"}),
                "sleeve": ("BOOLEAN", {"default": False, "label_on": "enabled(袖子)", "label_off": "disabled(袖子)"}),
                "pocket": ("BOOLEAN", {"default": False, "label_on": "enabled(口袋)", "label_off": "disabled(口袋)"}),
                "neckline": ("BOOLEAN", {"default": False, "label_on": "enabled(领口)", "label_off": "disabled(领口)"}),
                "buckle": ("BOOLEAN", {"default": False, "label_on": "enabled(带扣)", "label_off": "disabled(带扣)"}),
                "zipper": ("BOOLEAN", {"default": False, "label_on": "enabled(拉链)", "label_off": "disabled(拉链)"}),
                "applique": ("BOOLEAN", {"default": False, "label_on": "enabled(贴花)", "label_off": "disabled(贴花)"}),
                "bead": ("BOOLEAN", {"default": False, "label_on": "enabled(珠子)", "label_off": "disabled(珠子)"}),
                "bow": ("BOOLEAN", {"default": False, "label_on": "enabled(蝴蝶结)", "label_off": "disabled(蝴蝶结)"}),
                "flower": ("BOOLEAN", {"default": False, "label_on": "enabled(花)", "label_off": "disabled(花)"}),
                "fringe": ("BOOLEAN", {"default": False, "label_on": "enabled(刘海)", "label_off": "disabled(刘海)"}),
                "ribbon": ("BOOLEAN", {"default": False, "label_on": "enabled(丝带)", "label_off": "disabled(丝带)"}),
                "rivet": ("BOOLEAN", {"default": False, "label_on": "enabled(铆钉)", "label_off": "disabled(铆钉)"}),
                "ruffle": ("BOOLEAN", {"default": False, "label_on": "enabled(褶饰)", "label_off": "disabled(褶饰)"}),
                "sequin": ("BOOLEAN", {"default": False, "label_on": "enabled(亮片)", "label_off": "disabled(亮片)"}),
                "tassel": ("BOOLEAN", {"default": False, "label_on": "enabled(流苏)", "label_off": "disabled(流苏)"}),
            }
        }

    RETURN_TYPES = ("SegPipeline",)
    RETURN_NAMES = ("segformer_pipeline",)
    FUNCTION = "segformer_fashion_pipeline_loader"
    CATEGORY = '😺dzNodes/LayerMask'

    def segformer_fashion_pipeline_loader(self, model,
                                          shirt, top, sweater, cardigan, jacket, vest, pants,
                                          shorts, skirt, coat, dress, jumpsuit, cape, glasses,
                                          hat, hairaccessory, tie, glove, watch, belt, legwarmer,
                                          tights, sock, shoe, bagwallet, scarf, umbrella, hood,
                                          collar, lapel, epaulette, sleeve, pocket, neckline,
                                          buckle, zipper, applique, bead, bow, flower, fringe,
                                          ribbon, rivet, ruffle, sequin, tassel
                                        ):

        pipeline = SegformerPipeline()
        labels_to_keep = [0]
        if not shirt:
            labels_to_keep.append(1)
        if not top:
            labels_to_keep.append(2)
        if not sweater:
            labels_to_keep.append(3)
        if not cardigan:
            labels_to_keep.append(4)
        if not jacket:
            labels_to_keep.append(5)
        if not vest:
            labels_to_keep.append(6)
        if not pants:
            labels_to_keep.append(7)
        if not shorts:
            labels_to_keep.append(8)
        if not skirt:
            labels_to_keep.append(9)
        if not coat:
            labels_to_keep.append(10)
        if not dress:
            labels_to_keep.append(11)
        if not jumpsuit:
            labels_to_keep.append(12)
        if not cape:
            labels_to_keep.append(13)
        if not glasses:
            labels_to_keep.append(14)
        if not hat:
            labels_to_keep.append(15)
        if not hairaccessory:
            labels_to_keep.append(16)
        if not tie:
            labels_to_keep.append(17)
        if not glove:
            labels_to_keep.append(18)
        if not watch:
            labels_to_keep.append(19)
        if not belt:
            labels_to_keep.append(20)
        if not legwarmer:
            labels_to_keep.append(21)
        if not tights:
            labels_to_keep.append(22)
        if not sock:
            labels_to_keep.append(23)
        if not shoe:
            labels_to_keep.append(24)
        if not bagwallet:
            labels_to_keep.append(25)
        if not scarf:
            labels_to_keep.append(26)
        if not umbrella:
            labels_to_keep.append(27)
        if not hood:
            labels_to_keep.append(28)
        if not collar:
            labels_to_keep.append(29)
        if not lapel:
            labels_to_keep.append(30)
        if not epaulette:
            labels_to_keep.append(31)
        if not sleeve:
            labels_to_keep.append(32)
        if not pocket:
            labels_to_keep.append(33)
        if not neckline:
            labels_to_keep.append(34)
        if not buckle:
            labels_to_keep.append(35)
        if not zipper:
            labels_to_keep.append(36)
        if not applique:
            labels_to_keep.append(37)
        if not bead:
            labels_to_keep.append(38)
        if not bow:
            labels_to_keep.append(39)
        if not flower:
            labels_to_keep.append(40)
        if not fringe:
            labels_to_keep.append(41)
        if not ribbon:
            labels_to_keep.append(42)
        if not rivet:
            labels_to_keep.append(43)
        if not ruffle:
            labels_to_keep.append(44)
        if not sequin:
            labels_to_keep.append(45)
        if not tassel:
            labels_to_keep.append(46)

        pipeline.segment_label = labels_to_keep
        pipeline.model_name = model
        return (pipeline,)

class SegformerUltraV2:

    def __init__(self):
        self.NODE_NAME = 'SegformerUltraV2'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda', 'cpu']
        return {"required":
            {
                "image": ("IMAGE",),
                "segformer_pipeline": ("SegPipeline",),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 8, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "segformer_ultra_v2"
    CATEGORY = '😺dzNodes/LayerMask'

    def segformer_ultra_v2(self, image, segformer_pipeline,
                        detail_method, detail_erode, detail_dilate, black_point, white_point,
                        process_detail, device, max_megapixels,
                        ):
        model = segformer_pipeline.model_name
        labels_to_keep = segformer_pipeline.segment_label
        ret_images = []
        ret_masks = []

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        for i in image:
            pred_seg, cloth = get_segmentation(i, model_name=model)
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            orig_image = tensor2pil(i).convert('RGB')

            mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)

            # 创建agnostic-mask图像
            mask_image = Image.fromarray((1 - mask) * 255)
            mask_image = mask_image.convert("L")
            brightness_image = ImageEnhance.Brightness(mask_image)
            mask_image = brightness_image.enhance(factor=1.08)
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


class LS_SegformerClothesSetting:

    def __init__(self):
        self.NODE_NAME = 'SegformerClothesSetting'
        pass

    # Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes",
    # 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe",
    # 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
    #  17: "Scarf"

    @classmethod
    def INPUT_TYPES(cls):

        return {"required":
            {   "face": ("BOOLEAN", {"default": False, "label_on": "enabled(脸)", "label_off": "disabled(脸)"}),
                "hair": ("BOOLEAN", {"default": False, "label_on": "enabled(头发)", "label_off": "disabled(头发)"}),
                "hat": ("BOOLEAN", {"default": False, "label_on": "enabled(帽子)", "label_off": "disabled(帽子)"}),
                "sunglass": ("BOOLEAN", {"default": False, "label_on": "enabled(墨镜)", "label_off": "disabled(墨镜)"}),
                "left_arm": ("BOOLEAN", {"default": False, "label_on": "enabled(左臂)", "label_off": "disabled(左臂)"}),
                "right_arm": ("BOOLEAN", {"default": False, "label_on": "enabled(右臂)", "label_off": "disabled(右臂)"}),
                "left_leg": ("BOOLEAN", {"default": False, "label_on": "enabled(左腿)", "label_off": "disabled(左腿)"}),
                "right_leg": ("BOOLEAN", {"default": False, "label_on": "enabled(右腿)", "label_off": "disabled(右腿)"}),
                "left_shoe": ("BOOLEAN", {"default": False, "label_on": "enabled(左鞋)", "label_off": "disabled(左鞋)"}),
                "right_shoe": ("BOOLEAN", {"default": False, "label_on": "enabled(右鞋)", "label_off": "disabled(右鞋)"}),
                "upper_clothes": ("BOOLEAN", {"default": False, "label_on": "enabled(上衣)", "label_off": "disabled(上衣)"}),
                "skirt": ("BOOLEAN", {"default": False, "label_on": "enabled(短裙)", "label_off": "disabled(短裙)"}),
                "pants": ("BOOLEAN", {"default": False, "label_on": "enabled(裤子)", "label_off": "disabled(裤子)"}),
                "dress": ("BOOLEAN", {"default": False, "label_on": "enabled(连衣裙)", "label_off": "disabled(连衣裙)"}),
                "belt": ("BOOLEAN", {"default": False, "label_on": "enabled(腰带)", "label_off": "disabled(腰带)"}),
                "bag": ("BOOLEAN", {"default": False, "label_on": "enabled(背包)", "label_off": "disabled(背包)"}),
                "scarf": ("BOOLEAN", {"default": False, "label_on": "enabled(围巾)", "label_off": "disabled(围巾)"}),
            }
        }

    RETURN_TYPES = ("LS_SEGFORMER_SETTING",)
    RETURN_NAMES = ("segformer_clothes_setting",)
    FUNCTION = "run_segformer_clothes_setting"
    CATEGORY = '😺dzNodes/LayerMask'

    def run_segformer_clothes_setting(self, face, hat, hair, sunglass,
                        left_leg, right_leg, left_arm, right_arm, left_shoe, right_shoe,
                        upper_clothes, skirt, pants, dress, belt, bag, scarf,
                        ):

        pipeline = SegformerPipeline()
        labels_to_keep = [0]
        if not hat:
            labels_to_keep.append(1)
        if not hair:
            labels_to_keep.append(2)
        if not sunglass:
            labels_to_keep.append(3)
        if not upper_clothes:
            labels_to_keep.append(4)
        if not skirt:
            labels_to_keep.append(5)
        if not pants:
            labels_to_keep.append(6)
        if not dress:
            labels_to_keep.append(7)
        if not belt:
            labels_to_keep.append(8)
        if not left_shoe:
            labels_to_keep.append(9)
        if not right_shoe:
            labels_to_keep.append(10)
        if not face:
            labels_to_keep.append(11)
        if not left_leg:
            labels_to_keep.append(12)
        if not right_leg:
            labels_to_keep.append(13)
        if not left_arm:
            labels_to_keep.append(14)
        if not right_arm:
            labels_to_keep.append(15)
        if not bag:
            labels_to_keep.append(16)
        if not scarf:
            labels_to_keep.append(17)

        setting = {"labels_to_keep": labels_to_keep, "model_name": "segformer_b3_clothes"}

        return (setting,)

class LS_SegformerFashionSetting:

    def __init__(self):
        self.NODE_NAME = 'SegformerFashionSetting'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {   "shirt": ("BOOLEAN", {"default": False, "label_on": "enabled(衬衫、罩衫)", "label_off": "disabled(衬衫、罩衫)"}),
                "top": ("BOOLEAN", {"default": False, "label_on": "enabled(上衣、t恤)", "label_off": "disabled(上衣、t恤)"}),
                "sweater": ("BOOLEAN", {"default": False, "label_on": "enabled(毛衣)", "label_off": "disabled(毛衣)"}),
                "cardigan": ("BOOLEAN", {"default": False, "label_on": "enabled(开襟毛衫)", "label_off": "disabled(开襟毛衫)"}),
                "jacket": ("BOOLEAN", {"default": False, "label_on": "enabled(夹克)", "label_off": "disabled(夹克)"}),
                "vest": ("BOOLEAN", {"default": False, "label_on": "enabled(背心)", "label_off": "disabled(背心)"}),
                "pants": ("BOOLEAN", {"default": False, "label_on": "enabled(裤子)", "label_off": "disabled(裤子)"}),
                "shorts": ("BOOLEAN", {"default": False, "label_on": "enabled(短裤)", "label_off": "disabled(短裤)"}),
                "skirt": ("BOOLEAN", {"default": False, "label_on": "enabled(裙子)", "label_off": "disabled(裙子)"}),
                "coat": ("BOOLEAN", {"default": False, "label_on": "enabled(外套)", "label_off": "disabled(外套)"}),
                "dress": ("BOOLEAN", {"default": False, "label_on": "enabled(连衣裙)", "label_off": "disabled(连衣裙)"}),
                "jumpsuit": ("BOOLEAN", {"default": False, "label_on": "enabled(连身裤)", "label_off": "disabled(连身裤)"}),
                "cape": ("BOOLEAN", {"default": False, "label_on": "enabled(斗篷)", "label_off": "disabled(斗篷)"}),
                "glasses": ("BOOLEAN", {"default": False, "label_on": "enabled(眼镜)", "label_off": "disabled(眼镜)"}),
                "hat": ("BOOLEAN", {"default": False, "label_on": "enabled(帽子)", "label_off": "disabled(帽子)"}),
                "hairaccessory": ("BOOLEAN", {"default": False, "label_on": "enabled(头带)", "label_off": "disabled(头带)"}),
                "tie": ("BOOLEAN", {"default": False, "label_on": "enabled(领带)", "label_off": "disabled(领带)"}),
                "glove": ("BOOLEAN", {"default": False, "label_on": "enabled(手套)", "label_off": "disabled(手套)"}),
                "watch": ("BOOLEAN", {"default": False, "label_on": "enabled(手表)", "label_off": "disabled(手表)"}),
                "belt": ("BOOLEAN", {"default": False, "label_on": "enabled(皮带)", "label_off": "disabled(皮带)"}),
                "legwarmer": ("BOOLEAN", {"default": False, "label_on": "enabled(腿套)", "label_off": "disabled(腿套)"}),
                "tights": ("BOOLEAN", {"default": False, "label_on": "enabled(裤袜)","label_off": "disabled(裤袜)"}),
                "sock": ("BOOLEAN", {"default": False, "label_on": "enabled(袜子)", "label_off": "disabled(袜子)"}),
                "shoe": ("BOOLEAN", {"default": False, "label_on": "enabled(鞋子)", "label_off": "disabled(鞋子)"}),
                "bagwallet": ("BOOLEAN", {"default": False, "label_on": "enabled(手包)", "label_off": "disabled(手包)"}),
                "scarf": ("BOOLEAN", {"default": False, "label_on": "enabled(围巾)", "label_off": "disabled(围巾)"}),
                "umbrella": ("BOOLEAN", {"default": False, "label_on": "enabled(雨伞)", "label_off": "disabled(雨伞)"}),
                "hood": ("BOOLEAN", {"default": False, "label_on": "enabled(兜帽)", "label_off": "disabled(兜帽)"}),
                "collar": ("BOOLEAN", {"default": False, "label_on": "enabled(衣领)", "label_off": "disabled(衣领)"}),
                "lapel": ("BOOLEAN", {"default": False, "label_on": "enabled(翻领)", "label_off": "disabled(翻领)"}),
                "epaulette": ("BOOLEAN", {"default": False, "label_on": "enabled(肩章)", "label_off": "disabled(肩章)"}),
                "sleeve": ("BOOLEAN", {"default": False, "label_on": "enabled(袖子)", "label_off": "disabled(袖子)"}),
                "pocket": ("BOOLEAN", {"default": False, "label_on": "enabled(口袋)", "label_off": "disabled(口袋)"}),
                "neckline": ("BOOLEAN", {"default": False, "label_on": "enabled(领口)", "label_off": "disabled(领口)"}),
                "buckle": ("BOOLEAN", {"default": False, "label_on": "enabled(带扣)", "label_off": "disabled(带扣)"}),
                "zipper": ("BOOLEAN", {"default": False, "label_on": "enabled(拉链)", "label_off": "disabled(拉链)"}),
                "applique": ("BOOLEAN", {"default": False, "label_on": "enabled(贴花)", "label_off": "disabled(贴花)"}),
                "bead": ("BOOLEAN", {"default": False, "label_on": "enabled(珠子)", "label_off": "disabled(珠子)"}),
                "bow": ("BOOLEAN", {"default": False, "label_on": "enabled(蝴蝶结)", "label_off": "disabled(蝴蝶结)"}),
                "flower": ("BOOLEAN", {"default": False, "label_on": "enabled(花)", "label_off": "disabled(花)"}),
                "fringe": ("BOOLEAN", {"default": False, "label_on": "enabled(刘海)", "label_off": "disabled(刘海)"}),
                "ribbon": ("BOOLEAN", {"default": False, "label_on": "enabled(丝带)", "label_off": "disabled(丝带)"}),
                "rivet": ("BOOLEAN", {"default": False, "label_on": "enabled(铆钉)", "label_off": "disabled(铆钉)"}),
                "ruffle": ("BOOLEAN", {"default": False, "label_on": "enabled(褶饰)", "label_off": "disabled(褶饰)"}),
                "sequin": ("BOOLEAN", {"default": False, "label_on": "enabled(亮片)", "label_off": "disabled(亮片)"}),
                "tassel": ("BOOLEAN", {"default": False, "label_on": "enabled(流苏)", "label_off": "disabled(流苏)"}),
            }
        }

    RETURN_TYPES = ("LS_SEGFORMER_SETTING",)
    RETURN_NAMES = ("segformer_fashion_setting",)
    FUNCTION = "run_segformer_fashion_setting"
    CATEGORY = '😺dzNodes/LayerMask'

    def run_segformer_fashion_setting(self, shirt, top, sweater, cardigan, jacket, vest, pants,
                                      shorts, skirt, coat, dress, jumpsuit, cape, glasses,
                                      hat, hairaccessory, tie, glove, watch, belt, legwarmer,
                                      tights, sock, shoe, bagwallet, scarf, umbrella, hood,
                                      collar, lapel, epaulette, sleeve, pocket, neckline,
                                      buckle, zipper, applique, bead, bow, flower, fringe,
                                      ribbon, rivet, ruffle, sequin, tassel
                                     ):

        pipeline = SegformerPipeline()
        labels_to_keep = [0]
        if not shirt:
            labels_to_keep.append(1)
        if not top:
            labels_to_keep.append(2)
        if not sweater:
            labels_to_keep.append(3)
        if not cardigan:
            labels_to_keep.append(4)
        if not jacket:
            labels_to_keep.append(5)
        if not vest:
            labels_to_keep.append(6)
        if not pants:
            labels_to_keep.append(7)
        if not shorts:
            labels_to_keep.append(8)
        if not skirt:
            labels_to_keep.append(9)
        if not coat:
            labels_to_keep.append(10)
        if not dress:
            labels_to_keep.append(11)
        if not jumpsuit:
            labels_to_keep.append(12)
        if not cape:
            labels_to_keep.append(13)
        if not glasses:
            labels_to_keep.append(14)
        if not hat:
            labels_to_keep.append(15)
        if not hairaccessory:
            labels_to_keep.append(16)
        if not tie:
            labels_to_keep.append(17)
        if not glove:
            labels_to_keep.append(18)
        if not watch:
            labels_to_keep.append(19)
        if not belt:
            labels_to_keep.append(20)
        if not legwarmer:
            labels_to_keep.append(21)
        if not tights:
            labels_to_keep.append(22)
        if not sock:
            labels_to_keep.append(23)
        if not shoe:
            labels_to_keep.append(24)
        if not bagwallet:
            labels_to_keep.append(25)
        if not scarf:
            labels_to_keep.append(26)
        if not umbrella:
            labels_to_keep.append(27)
        if not hood:
            labels_to_keep.append(28)
        if not collar:
            labels_to_keep.append(29)
        if not lapel:
            labels_to_keep.append(30)
        if not epaulette:
            labels_to_keep.append(31)
        if not sleeve:
            labels_to_keep.append(32)
        if not pocket:
            labels_to_keep.append(33)
        if not neckline:
            labels_to_keep.append(34)
        if not buckle:
            labels_to_keep.append(35)
        if not zipper:
            labels_to_keep.append(36)
        if not applique:
            labels_to_keep.append(37)
        if not bead:
            labels_to_keep.append(38)
        if not bow:
            labels_to_keep.append(39)
        if not flower:
            labels_to_keep.append(40)
        if not fringe:
            labels_to_keep.append(41)
        if not ribbon:
            labels_to_keep.append(42)
        if not rivet:
            labels_to_keep.append(43)
        if not ruffle:
            labels_to_keep.append(44)
        if not sequin:
            labels_to_keep.append(45)
        if not tassel:
            labels_to_keep.append(46)

        setting = {"labels_to_keep":labels_to_keep, "model_name":"segformer_b3_fashion"}

        return (setting,)

class LS_LoadSegformerModel:

    def __init__(self):
        self.NODE_NAME = 'LoadSegformerModel'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        model_list = ['segformer_b3_clothes', 'segformer_b2_clothes', 'segformer_b3_fashion']
        device_list = ['cuda', 'cpu']
        return {"required":
                   {
                    "model_name": (model_list,),
                    "device": (device_list,),
                   }
               }

    RETURN_TYPES = ("LS_SEGFORMER_MODEL", )
    RETURN_NAMES = ("segfromer_model", )
    FUNCTION = "load_segformer_model"
    CATEGORY = '😺dzNodes/LayerMask'

    def load_segformer_model(self, model_name, device):

        model_folder_path = os.path.join(folder_paths.models_dir, model_name)
        try:
            model_folder_path = os.path.normpath(folder_paths.folder_names_and_paths[model_name][0][0])
        except:
            pass

        processor = SegformerImageProcessor.from_pretrained(model_folder_path)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_folder_path)

        segfromer_model = {"processor":processor, "model":model, "device":device, "model_name":model_name}

        log(f"{self.NODE_NAME} Loaded Segformer Model {model_name}.", message_type='finish')
        return (segfromer_model,)

class LS_SegformerUltraV3:

    def __init__(self):
        self.NODE_NAME = 'SegformerUltraV3'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        return {"required":
                    {
                        "image": ("IMAGE",),
                        "segformer_model": ("LS_SEGFORMER_MODEL",),
                        "segformer_setting": ("LS_SEGFORMER_SETTING",),
                        "detail_method": (method_list,),
                        "detail_erode": ("INT", {"default": 8, "min": 1, "max": 255, "step": 1}),
                        "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                        "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                        "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                        "process_detail": ("BOOLEAN", {"default": True}),
                        "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
                    }
                }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "segformer_ultra_v3"
    CATEGORY = '😺dzNodes/LayerMask'

    def segformer_ultra_v3(self, image, segformer_model, segformer_setting,
                        detail_method, detail_erode, detail_dilate, black_point, white_point,
                        process_detail, max_megapixels,
                        ):

        device = segformer_model["device"]
        model_name = segformer_model["model_name"]

        labels_to_keep = segformer_setting["labels_to_keep"]
        labels_model_name = segformer_setting["model_name"]

        ret_images = []
        ret_masks = []

        if model_name.rsplit('_', 1)[-1] != labels_model_name.rsplit('_', 1)[-1]: # 后缀不一致
            raise TypeError("Segformer Model and Segformer Setting are different.")

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        for i in image:
            pred_seg, cloth = get_segmentation_from_model(i, segformer_model)
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            orig_image = tensor2pil(i).convert('RGB')

            mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)

            # 创建agnostic-mask图像
            mask_image = Image.fromarray((1 - mask) * 255)
            mask_image = mask_image.convert("L")
            brightness_image = ImageEnhance.Brightness(mask_image)
            mask_image = brightness_image.enhance(factor=1.08)
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
    "LayerMask: SegformerB2ClothesUltra": Segformer_B2_Clothes,
    "LayerMask: SegformerUltraV2": SegformerUltraV2,
    "LayerMask: SegformerClothesPipelineLoader": SegformerClothesPipelineLoader,
    "LayerMask: SegformerFashionPipelineLoader": SegformerFashionPipelineLoader,
    "LayerMask: SegformerUltraV3": LS_SegformerUltraV3,
    "LayerMask: SegformerClothesSetting": LS_SegformerClothesSetting,
    "LayerMask: SegformerFashionSetting": LS_SegformerFashionSetting,
    "LayerMask: LoadSegformerModel": LS_LoadSegformerModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: SegformerB2ClothesUltra": "LayerMask: Segformer B2 Clothes Ultra",
    "LayerMask: SegformerUltraV2": "LayerMask: Segformer Ultra V2",
    "LayerMask: SegformerClothesPipelineLoader": "LayerMask: Segformer Clothes Pipeline",
    "LayerMask: SegformerFashionPipelineLoader": "LayerMask: Segformer Fashion Pipeline",
    "LayerMask: SegformerUltraV3": "LayerMask: Segformer Ultra V3",
    "LayerMask: SegformerClothesSetting": "LayerMask: Segformer Clothes Setting",
    "LayerMask: SegformerFashionSetting": "LayerMask: Segformer Fashion Setting",
    "LayerMask: LoadSegformerModel": "LayerMask: Load Segformer Model",
}

