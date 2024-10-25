import os
import sys
import torch
import re
from .imagefunc import *
from transformers import pipeline
import folder_paths

vqa_model_path = os.path.join(folder_paths.models_dir, 'VQA')

vqa_model_repos = {
    "blip-vqa-base": "Salesforce/blip-vqa-base",
    "blip-vqa-capfilt-large": "Salesforce/blip-vqa-capfilt-large",
}

def get_models():
    sub_dirs = []
    for  filename in os.listdir(vqa_model_path):
        if os.path.isdir(os.path.join(vqa_model_path, filename)):
            sub_dirs.append(filename)
    return sub_dirs

class LS_LoadVQAModel:

    def __init__(self):
        self.processor = None
        self.model = None
        self.model_name = ""
        self.device = ""
        self.precision = ""

    @classmethod
    def INPUT_TYPES(s):
        model_list = list(vqa_model_repos.keys())
        precision_list = ["fp16", "fp32"]
        device_list = ['cuda','cpu']
        return {
            "required": {
                "model": (model_list,),
                "precision": (precision_list,),
                "device": (device_list,),
            },
        }

    RETURN_TYPES = ("VQA_MODEL",)
    RETURN_NAMES = ("vqa_model",)
    FUNCTION = "load_vqa_model"
    CATEGORY = '😺dzNodes/LayerUtility'

    def load_vqa_model(self, model, precision, device):

        if (model == self.model_name and precision == self.precision and device == self.device
                and self.model is not None and self.processor is not None):
            return ([self.processor, self.model, device, precision, self.model_name],)

        model_path = os.path.join(vqa_model_path, model)
        from transformers import BlipProcessor,BlipForQuestionAnswering

        # if there is no local files, use repo id to auto-download the dependencies. 
        if not os.path.exists(model_path):
            model_path = vqa_model_repos[model]
            
        vqa_processor = BlipProcessor.from_pretrained(model_path)
        if precision == 'fp16':
            vqa_model = BlipForQuestionAnswering.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        else:
            vqa_model = BlipForQuestionAnswering.from_pretrained(model_path).to(device)

        self.processor = vqa_processor
        self.model = vqa_model
        self.model_name = model
        self.device = device
        self.precision = precision

        return ([vqa_processor, vqa_model, device, precision, model],)

class LS_VQA_Prompt:

    def __init__(self):
        self.NODE_NAME = 'VQA Prompt'

    @classmethod
    def INPUT_TYPES(cls):
        default_question = "{age number} years old {ethnicity} {gender}, weared {garment color} {garment}, {eye color} eyes, {hair style} {hair color} hair, {background} background."

        return {
            "required": {
                "image": ("IMAGE",),
                "vqa_model": ("VQA_MODEL",),
                "question": ("STRING", {"default": default_question, "multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "vqa_prompt"
    CATEGORY = '😺dzNodes/LayerUtility'
  
    def vqa_prompt(self, image, vqa_model, question):
        answers = []
        [vqa_processor, vqa_model, device, precision, model_name] = vqa_model

        for img in image:
            _img = tensor2pil(img).convert("RGB")
            final_answer = question
            matches = re.findall(r'\{([^}]*)\}', question)

            for match in matches:
                if precision == 'fp16':
                    inputs = vqa_processor(_img, match, return_tensors="pt").to(device, torch.float16)
                else:
                    inputs = vqa_processor(_img, match, return_tensors="pt").to(device)
                out = vqa_model.generate(**inputs)
                match_answer = vqa_processor.decode(out[0], skip_special_tokens=True)
                log(f'{self.NODE_NAME} Q:"{match}", A:"{match_answer}"')
                final_answer = final_answer.replace("{" + match + "}", match_answer)
            answers.append(final_answer)

        log(f"{self.NODE_NAME} Processed.", message_type='finish')
        return (answers,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: VQAPrompt": LS_VQA_Prompt,
    "LayerUtility: LoadVQAModel": LS_LoadVQAModel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: VQAPrompt": "LayerUtility: VQA Prompt",
    "LayerUtility: LoadVQAModel": "LayerUtility: Load VQA Model"
}
