import os.path
from pathlib import Path
from transformers import AutoModel, AutoProcessor, StoppingCriteria, StoppingCriteriaList
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from huggingface_hub import snapshot_download
import folder_paths

# Define the directory for saving files related to uform-gen2-qwen
# files_for_uform_gen2_qwen = Path(folder_paths.folder_names_and_paths["LLavacheckpoints"][0][0]) / "files_for_uform_gen2_qwen"
files_for_uform_gen2_qwen = Path(os.path.join(folder_paths.models_dir, "LLavacheckpoints", "files_for_uform_gen2_qwen"))
files_for_uform_gen2_qwen.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [151645]  # Define stop tokens as per your model's specifics
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class UformGen2QwenChat:
    def __init__(self):
        self.model_path = snapshot_download("unum-cloud/uform-gen2-qwen-500m", 
                                            local_dir=files_for_uform_gen2_qwen,
                                            force_download=False,  # Set to True if you always want to download, regardless of local copy
                                            local_files_only=False,  # Set to False to allow downloading if not available locally
                                            local_dir_use_symlinks="auto") # or set to True/False based on your symlink preference
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

    def chat_response(self, message, history, image_path):
        stop = StopOnTokens()
        messages = [{"role": "system", "content": "You are a helpful Assistant."}]

        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        if len(messages) == 1:
            message = f" <image>{message}"

        messages.append({"role": "user", "content": message})

        model_inputs = self.processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        image = Image.open(image_path)  # Load image using PIL
        image_tensor = (
            self.processor.feature_extractor(image)
            .unsqueeze(0)
        )

        attention_mask = torch.ones(
            1, model_inputs.shape[1] + self.processor.num_image_latents - 1
        )

        model_inputs = {
            "input_ids": model_inputs,
            "images": image_tensor,
            "attention_mask": attention_mask
        }

        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        output = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            stopping_criteria=StoppingCriteriaList([stop])
        )

        response_text = self.processor.tokenizer.decode(output[0], skip_special_tokens=True)
        return response_text

# Example of integrating UformGen2QwenChat into a node-like structure
class QWenImage2Prompt:
    def __init__(self):
        self.chat_model = UformGen2QwenChat()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": ("STRING", {"multiline": False, "default": "describe this image",},),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "uform_gen2_qwen_chat"
    CATEGORY = '😺dzNodes/LayerUtility/Prompt'

    def uform_gen2_qwen_chat(self, image, question):
        history = []  # Example empty history
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))
        temp_path = files_for_uform_gen2_qwen / "temp.png"
        pil_image.save(temp_path)
        
        response = self.chat_model.chat_response(question, history, temp_path)
        return (response.split("assistant\n", 1)[1], )

NODE_CLASS_MAPPINGS = {
    "LayerUtility: QWenImage2Prompt": QWenImage2Prompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: QWenImage2Prompt": "LayerUtility: QWenImage2Prompt"
}