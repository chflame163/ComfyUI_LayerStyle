# Based on https://github.com/SeanScripts/ComfyUI-PixtralLlamaMolmoVision
import os
import comfy.model_management as mm
import folder_paths

from .imagefunc import tensor2pil, log, clear_memory

class LS_LlamaVision:

    def __init__(self):
        self.NODE_NAME = 'Llama Vision'
        self.previous_model = None

    @classmethod
    def INPUT_TYPES(s):
        model_list = ["Llama-3.2-11B-Vision-Instruct-nf4"]
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (model_list,),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant.", "multiline": True}),
                "user_prompt": ("STRING", {"default": "Describe this image in natural language.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 4096}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1}),
                "top_k": ("INT", {"default": 40, "min": 1}),
                "stop_strings": ("STRING", {"default": "<|eot_id|>"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "include_prompt_in_output": ("BOOLEAN", {"default": False}),
                "cache_model": ("BOOLEAN", {"default": False}),
            },
            "optional": {
            },
        }

    CATEGORY = '😺dzNodes/LayerUtility'
    FUNCTION = "llama_vision"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (True,)

    def llama_vision(self, image, model, system_prompt, user_prompt, max_new_tokens, do_sample, temperature,
                      top_p, top_k, stop_strings, seed, include_prompt_in_output, cache_model,):

        from transformers import MllamaForConditionalGeneration, AutoProcessor, GenerationConfig, StopStringCriteria, set_seed

        device = mm.get_torch_device()
        if self.previous_model is not None:
            llama_vision_model = self.previous_model
        else:
            model_path = os.path.join(folder_paths.models_dir, 'LLM', model)
            # Don't load the full model until needed for generation
            processor = AutoProcessor.from_pretrained(model_path)
            llama_vision_model = {
                'path': model_path,
                'processor': processor,
            }

        if llama_vision_model['path'] and 'model' not in llama_vision_model:
            llama_vision_model['model'] = MllamaForConditionalGeneration.from_pretrained(
                llama_vision_model['path'],
                use_safetensors=True,
                device_map=device,
            )

        ret_texts = []

        for img in image:
            img = tensor2pil(img.unsqueeze(0))
            # Process prompt
            image_tags = "<|image|>" * len(image)
            final_prompt = "<|begin_of_text|>"
            if system_prompt != "":
                final_prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n\n"
            final_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{image_tags}{user_prompt}<|eot_id|>\n\n"
            final_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

            inputs = llama_vision_model['processor'](images=[img], text=final_prompt, return_tensors="pt").to(device)
            prompt_tokens = len(inputs['input_ids'][0])
            stop_strings_list = stop_strings.split(",")
            set_seed(seed)
            generate_ids = llama_vision_model['model'].generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ),
                stopping_criteria=[StopStringCriteria(tokenizer=llama_vision_model['processor'].tokenizer,
                                                      stop_strings=stop_strings_list)],
            )

            generated_tokens = len(generate_ids[0]) - prompt_tokens
            output_tokens = generate_ids[0] if include_prompt_in_output else generate_ids[0][prompt_tokens:]
            output = llama_vision_model['processor'].decode(output_tokens, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)
            log(f"{self.NODE_NAME} generated: {output}")
            ret_texts.append(output)

        if cache_model:
            self.previous_model = llama_vision_model
        else:
            self.previous_model = None
            del llama_vision_model
            clear_memory()
        log(f"{self.NODE_NAME} generated {len(ret_texts)} texts.")
        return (ret_texts,)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: LlamaVision": LS_LlamaVision
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: LlamaVision": "LayerUtility: Llama Vision"
}