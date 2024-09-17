import numpy as np
import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, pipeline
import folder_paths
from .imagefunc import log, clear_memory

model_path = os.path.join(folder_paths.models_dir, 'LLM')

class LS_PhiModel:
    def __init__(self, name, device, dtype):
        self.name = name
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer= None
        self.processor = None

class LS_Phi_Prompt:

    CATEGORY = '😺dzNodes/LayerUtility'
    FUNCTION = "phi_prompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    def __init__(self):
        self.NODE_NAME = 'Phi Prompt'
        self.previous_model = LS_PhiModel("", "", "")

    @classmethod
    def INPUT_TYPES(self):
        phi_model_list = ["auto", "Phi-3.5-mini-instruct", "Phi-3.5-vision-instruct"]
        device_list = ['cuda', 'cpu']
        dtype_list = ['fp16', 'bf16', 'fp32']
        return {
            "required": {
                "model": (phi_model_list,),
                "device": (device_list,),
                "dtype": (dtype_list,),
                "cache_model": ("BOOLEAN", {"default": False}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant.","multiline": False}),
                "user_prompt": ("STRING", {"default": "Describe this image","multiline": True}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.01, "max":1, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 512,"min": 8, "max":4096, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    def phi_prompt(self, model, device, dtype, cache_model,
                   system_prompt, user_prompt, do_sample,
                   temperature, max_new_tokens, image=None):

        if model == "Phi-3.5-mini-instruct" or (model=="auto" and image is None):

            if (self.previous_model.name != "Phi-3.5-mini-instruct"
                    or self.previous_model.device != device
                    or self.previous_model.dtype != dtype):
                phi_model = self.load_phi_model("Phi-3.5-mini-instruct", device, dtype)
            else:
                phi_model = self.previous_model

            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Build pipeline
            pipe = pipeline("text-generation", model=phi_model.model, tokenizer=phi_model.tokenizer)
            generation_args = {
                "return_full_text": False,
                "do_sample": do_sample,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens
            }

            # Generate
            output = pipe(messages, **generation_args)
            response = output[0]["generated_text"]

        elif model == "Phi-3.5-vision-instruct" or (model=="auto" and image is not None):

            if image is None:
                log(f"{self.NODE_NAME} input is vision model but image is None.", message_type="error")
                return ("",)
            else:
                if (self.previous_model.name != "Phi-3.5-vision-instruct"
                        or self.previous_model.device != device
                        or self.previous_model.dtype != dtype):
                    phi_model = self.load_phi_model("Phi-3.5-vision-instruct", device, dtype)
                else:
                    phi_model = self.previous_model
                images = self.tensor2batch_pil(image) # Convert tensor to PIL image batch

                # Prepare images placeholders in the prompt
                placeholder = ''
                for index, value in enumerate(images, start=1):
                    placeholder += f"<|image_{index}|>\n"

                # Prepare prompt
                messages = [{"role": "user", "content": placeholder + user_prompt}]
                prompt = phi_model.processor.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Prepare generation arguments
                inputs = phi_model.processor(prompt, images, return_tensors="pt").to(device)
                generate_args = {}
                if do_sample:
                    generate_args["do_sample"] = do_sample
                    generate_args["temperature"] = temperature
                else:
                    generate_args["do_sample"] = do_sample

                # Generate
                generate_ids = phi_model.model.generate(
                    **inputs,
                    eos_token_id=phi_model.processor.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    **generate_args
                )

                # Remove input tokens
                generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                response = phi_model.processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

        log(f"{self.NODE_NAME} processed successfully.", message_type="finish")

        if cache_model:
            self.previous_model = phi_model
        else:
            self.previous_model = LS_PhiModel("", "", "")
            del phi_model
            clear_memory()
        response = response.strip()
        return (response,)

    def load_phi_model(self, model, device, dtype):
        phi_model =LS_PhiModel(model, device, dtype)
        model_dir = os.path.join(model_path, model)
        if dtype == 'fp16':
            torch_dtype = torch.float16
        elif dtype == 'bf16':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        if model == "Phi-3.5-mini-instruct":
            try:
                phi_model.model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_dir,
                    device_map=device,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
                phi_model.tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                )
            except Exception as e:
                log(f"{self.NODE_NAME} failed to load {model}. Error: {e}", message_type="error")

        elif model == "Phi-3.5-vision-instruct":
            try:
                phi_model.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map=device,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    # _attn_implementation="flash_attention_2",
                    _attn_implementation="eager"
                )
                # For best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
                phi_model.processor = AutoProcessor.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    num_crops=16
                )
            except Exception as e:
                log(f"{self.NODE_NAME} failed to load {model}. Error: {e}", message_type="error")

        return phi_model
    def tensor2batch_pil(self, image):
        batch_count = image.size(0) if len(image.shape) > 3 else 1
        if batch_count > 1:
            out = []
            for i in range(batch_count):
                out.extend(self.tensor2pil(image[i]))
            return out
        return [Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))]

NODE_CLASS_MAPPINGS = {
    "LayerUtility: PhiPrompt": LS_Phi_Prompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: PhiPrompt": "LayerUtility: Phi Prompt"
}
