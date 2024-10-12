# Based on https://huggingface.co/John6666/joy-caption-alpha-two-cli-mod

import os
import sys
import torch
import torch.amp.autocast_mode
from torch import nn
from typing import List, Union
from PIL import Image

import folder_paths
from .imagefunc import download_hg_model, log, tensor2pil, clear_memory

class Joy2_Model():
    def __init__(self, clip_processor, clip_model, tokenizer, text_model, image_adapter):
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.tokenizer =  tokenizer
        self.text_model = text_model
        self.image_adapter = image_adapter

class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int,
                 deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))

        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)  # Matches HF's implementation of llama3

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.concat((
                vision_outputs[-2],
                vision_outputs[3],
                vision_outputs[7],
                vision_outputs[13],
                vision_outputs[20],
            ), dim=-1)
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
            assert x.shape[-1] == vision_outputs[-2].shape[
                -1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        other_tokens = self.other_tokens(
            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
        assert other_tokens.shape == (
        x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

def load_models(model_path, dtype, vlm_lora, device):
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, \
        AutoModelForCausalLM
    from peft import PeftModel

    use_lora = True if vlm_lora != "none" else False
    CLIP_PATH = download_hg_model("google/siglip-so400m-patch14-384", "clip")
    CHECKPOINT_PATH = os.path.join(folder_paths.models_dir, "Joy_caption", "cgrkzexw-599808")
    LORA_PATH = os.path.join(CHECKPOINT_PATH, "text_model")

    try:
        if dtype=="nf4":
            from transformers import BitsAndBytesConfig
            nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            print("Loading in NF4")
            print("Loading CLIP 📎")
            clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
            clip_model = AutoModel.from_pretrained(CLIP_PATH).vision_model

            print("Loading VLM's custom vision model 📎")
            checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, "clip_model.pt"), map_location='cpu', weights_only=False)
            checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
            clip_model.load_state_dict(checkpoint)
            del checkpoint
            clip_model.eval().requires_grad_(False).to(device)

            print("Loading tokenizer 🪙")
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(CHECKPOINT_PATH, "text_model"), use_fast=True)
            assert isinstance(tokenizer,
                              (PreTrainedTokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(tokenizer)}"

            print(f"Loading LLM: {model_path} 🤖")
            text_model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config,
                                                              device_map=device, torch_dtype=torch.bfloat16).eval()

            if False and use_lora and os.path.exists(LORA_PATH):  # omitted
                print("Loading VLM's custom text model 🤖")
                text_model = PeftModel.from_pretrained(model=text_model, model_id=LORA_PATH, device_map=device,
                                                       quantization_config=nf4_config)
                text_model = text_model.merge_and_unload(
                    safe_merge=True)  # to avoid PEFT bug https://github.com/huggingface/transformers/issues/28515
            else:
                print("VLM's custom text model isn't loaded 🤖")

            print("Loading image adapter 🖼️")
            image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38,
                                         False).eval().to("cpu")
            image_adapter.load_state_dict(
                torch.load(os.path.join(CHECKPOINT_PATH, "image_adapter.pt"), map_location=device, weights_only=False))
            image_adapter.eval().to(device)
        else: # bf16
            print("Loading in bfloat16")
            print("Loading CLIP 📎")
            clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
            clip_model = AutoModel.from_pretrained(CLIP_PATH).vision_model
            if os.path.exists(os.path.join(CHECKPOINT_PATH, "clip_model.pt")):
                print("Loading VLM's custom vision model 📎")
                checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, "clip_model.pt"), map_location=device, weights_only=False)
                checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
                clip_model.load_state_dict(checkpoint)
                del checkpoint
            clip_model.eval().requires_grad_(False).to(device)

            print("Loading tokenizer 🪙")
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(CHECKPOINT_PATH, "text_model"), use_fast=True)
            assert isinstance(tokenizer,
                              (PreTrainedTokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(tokenizer)}"

            print(f"Loading LLM: {model_path} 🤖")
            text_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",
                                                              torch_dtype=torch.bfloat16).eval()  # device_map="auto" may cause LoRA issue

            if use_lora and os.path.exists(LORA_PATH):
                print("Loading VLM's custom text model 🤖")
                text_model = PeftModel.from_pretrained(model=text_model, model_id=LORA_PATH, device_map=device)
                text_model = text_model.merge_and_unload(
                    safe_merge=True)  # to avoid PEFT bug https://github.com/huggingface/transformers/issues/28515
            else:
                print("VLM's custom text model isn't loaded 🤖")

            print("Loading image adapter 🖼️")
            image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38,
                                         False).eval().to(device)
            image_adapter.load_state_dict(
                torch.load(os.path.join(CHECKPOINT_PATH, "image_adapter.pt"), map_location=device, weights_only=False))
    except Exception as e:
        print(f"Error loading models: {e}")
    finally:
        clear_memory()

    return Joy2_Model(clip_processor, clip_model, tokenizer, text_model, image_adapter)

@torch.inference_mode()
def stream_chat(input_images: List[Image.Image], caption_type: str, caption_length: Union[str, int],
                extra_options: list[str], name_input: str, custom_prompt: str,
                max_new_tokens: int, top_p: float, temperature: float, batch_size: int, model:Joy2_Model, device=str):

    CAPTION_TYPE_MAP = {
        "Descriptive": [
            "Write a descriptive caption for this image in a formal tone.",
            "Write a descriptive caption for this image in a formal tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a formal tone.",
        ],
        "Descriptive (Informal)": [
            "Write a descriptive caption for this image in a casual tone.",
            "Write a descriptive caption for this image in a casual tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a casual tone.",
        ],
        "Training Prompt": [
            "Write a stable diffusion prompt for this image.",
            "Write a stable diffusion prompt for this image within {word_count} words.",
            "Write a {length} stable diffusion prompt for this image.",
        ],
        "MidJourney": [
            "Write a MidJourney prompt for this image.",
            "Write a MidJourney prompt for this image within {word_count} words.",
            "Write a {length} MidJourney prompt for this image.",
        ],
        "Booru tag list": [
            "Write a list of Booru tags for this image.",
            "Write a list of Booru tags for this image within {word_count} words.",
            "Write a {length} list of Booru tags for this image.",
        ],
        "Booru-like tag list": [
            "Write a list of Booru-like tags for this image.",
            "Write a list of Booru-like tags for this image within {word_count} words.",
            "Write a {length} list of Booru-like tags for this image.",
        ],
        "Art Critic": [
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
        ],
        "Product Listing": [
            "Write a caption for this image as though it were a product listing.",
            "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
            "Write a {length} caption for this image as though it were a product listing.",
        ],
        "Social Media Post": [
            "Write a caption for this image as if it were being used for a social media post.",
            "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
            "Write a {length} caption for this image as if it were being used for a social media post.",
        ],
    }

    clear_memory()
    all_captions = []

    # 'any' means no length specified
    length = None if caption_length == "any" else caption_length

    if isinstance(length, str):
        try:
            length = int(length)
        except ValueError:
            pass

    # Build prompt
    if length is None:
        map_idx = 0
    elif isinstance(length, int):
        map_idx = 1
    elif isinstance(length, str):
        map_idx = 2
    else:
        raise ValueError(f"Invalid caption length: {length}")

    prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

    # Add extra options
    if len(extra_options) > 0:
        prompt_str += " " + " ".join(extra_options)

    # Add name, length, word_count
    prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()

    # For debugging
    print(f"Prompt: {prompt_str}")
    import torchvision.transforms.functional as TVF

    for i in range(0, len(input_images), batch_size):
        batch = input_images[i:i + batch_size]

        for input_image in input_images:
            try:
                # Preprocess image
                image = input_image.resize((384, 384), Image.LANCZOS)
                pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
                pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
                pixel_values = pixel_values.to(device)
            except ValueError as e:
                print(f"Error processing image: {e}")
                print("Skipping this image and continuing...")
                continue

            # Embed image
            # This results in Batch x Image Tokens x Features
            with torch.amp.autocast_mode.autocast(device, enabled=True):
                vision_outputs = model.clip_model(pixel_values=pixel_values, output_hidden_states=True)
                image_features = vision_outputs.hidden_states
                embedded_images = model.image_adapter(image_features).to(device)

            # Build the conversation
            convo = [
                {
                    "role": "system",
                    "content": "You are a helpful image captioner.",
                },
                {
                    "role": "user",
                    "content": prompt_str,
                },
            ]

            # Format the conversation
            convo_string = model.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            assert isinstance(convo_string, str)

            # Tokenize the conversation
            # prompt_str is tokenized separately so we can do the calculations below
            convo_tokens = model.tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False,
                                            truncation=False)
            prompt_tokens = model.tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False,
                                             truncation=False)
            assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
            convo_tokens = convo_tokens.squeeze(0)  # Squeeze just to make the following easier
            prompt_tokens = prompt_tokens.squeeze(0)

            # Calculate where to inject the image
            eot_id_indices = (convo_tokens == model.tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[
                0].tolist()
            assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"

            preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]  # Number of tokens before the prompt

            # Embed the tokens
            convo_embeds = model.text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(device))

            # Construct the input
            input_embeds = torch.cat([
                convo_embeds[:, :preamble_len],  # Part before the prompt
                embedded_images.to(dtype=convo_embeds.dtype),  # Image
                convo_embeds[:, preamble_len:],  # The prompt and anything after it
            ], dim=1).to(device)

            input_ids = torch.cat([
                convo_tokens[:preamble_len].unsqueeze(0),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                convo_tokens[preamble_len:].unsqueeze(0),
            ], dim=1).to(device)
            attention_mask = torch.ones_like(input_ids)

            generate_ids = model.text_model.generate(input_ids=input_ids, inputs_embeds=input_embeds,
                                               attention_mask=attention_mask, do_sample=True,
                                               suppress_tokens=None, max_new_tokens=max_new_tokens, top_p=top_p,
                                               temperature=temperature)

            # Trim off the prompt
            generate_ids = generate_ids[:, input_ids.shape[1]:]
            if generate_ids[0][-1] == model.tokenizer.eos_token_id or generate_ids[0][-1] == model.tokenizer.convert_tokens_to_ids(
                    "<|eot_id|>"):
                generate_ids = generate_ids[:, :-1]

            caption = model.tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
            all_captions.append(caption.strip())

    return all_captions


class LS_JoyCaptionExtraOptions:

    CATEGORY = '😺dzNodes/LayerUtility'
    FUNCTION = "extra_choice"
    RETURN_TYPES = ("JoyCaption2ExtraOption",)
    RETURN_NAMES = ("extra_option",)

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "refer_character_name": ("BOOLEAN", {"default": False}),
                "exclude_people_info": ("BOOLEAN", {"default": False}),
                "include_lighting": ("BOOLEAN", {"default": False}),
                "include_camera_angle": ("BOOLEAN", {"default": False}),
                "include_watermark": ("BOOLEAN", {"default": False}),
                "include_JPEG_artifacts": ("BOOLEAN", {"default": False}),
                "include_exif": ("BOOLEAN", {"default": False}),
                "exclude_sexual": ("BOOLEAN", {"default": False}),
                "exclude_image_resolution": ("BOOLEAN", {"default": False}),
                "include_aesthetic_quality": ("BOOLEAN", {"default": False}),
                "include_composition_style": ("BOOLEAN", {"default": False}),
                "exclude_text": ("BOOLEAN", {"default": False}),
                "specify_depth_field": ("BOOLEAN", {"default": False}),
                "specify_lighting_sources": ("BOOLEAN", {"default": False}),
                "do_not_use_ambiguous_language": ("BOOLEAN", {"default": False}),
                "include_nsfw": ("BOOLEAN", {"default": False}),
                "only_describe_most_important_elements": ("BOOLEAN", {"default": False}),
                "character_name": ("STRING", {"default": "Huluwa", "multiline": False}),
            },
            "optional": {
            }
        }

    def extra_choice(self, refer_character_name, exclude_people_info, include_lighting, include_camera_angle,
                     include_watermark, include_JPEG_artifacts, include_exif, exclude_sexual,
                     exclude_image_resolution, include_aesthetic_quality, include_composition_style,
                     exclude_text, specify_depth_field, specify_lighting_sources,
                     do_not_use_ambiguous_language, include_nsfw, only_describe_most_important_elements,
                     character_name):

        extra_list = {
            "refer_character_name":"If there is a person/character in the image you must refer to them as {name}.",
            "exclude_people_info":"Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
            "include_lighting":"Include information about lighting.",
            "include_camera_angle":"Include information about camera angle.",
            "include_watermark":"Include information about whether there is a watermark or not.",
            "include_JPEG_artifacts":"Include information about whether there are JPEG artifacts or not.",
            "include_exif":"If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
            "exclude_sexual":"Do NOT include anything sexual; keep it PG.",
            "exclude_image_resolution":"Do NOT mention the image's resolution.",
            "include_aesthetic_quality":"You MUST include information about the subjective aesthetic quality of the image from low to very high.",
            "include_composition_style":"Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
            "exclude_text":"Do NOT mention any text that is in the image.",
            "specify_depth_field":"Specify the depth of field and whether the background is in focus or blurred.",
            "specify_lighting_sources":"If applicable, mention the likely use of artificial or natural lighting sources.",
            "do_not_use_ambiguous_language":"Do NOT use any ambiguous language.",
            "include_nsfw":"Include whether the image is sfw, suggestive, or nsfw.",
            "only_describe_most_important_elements":"ONLY describe the most important elements of the image."
        }
        ret_list = []
        if refer_character_name:
            ret_list.append(extra_list["refer_character_name"])
        if exclude_people_info:
            ret_list.append(extra_list["exclude_people_info"])
        if include_lighting:
            ret_list.append(extra_list["include_lighting"])
        if include_camera_angle:
            ret_list.append(extra_list["include_camera_angle"])
        if include_watermark:
            ret_list.append(extra_list["include_watermark"])
        if include_JPEG_artifacts:
            ret_list.append(extra_list["include_JPEG_artifacts"])
        if include_exif:
            ret_list.append(extra_list["include_exif"])
        if exclude_sexual:
            ret_list.append(extra_list["exclude_sexual"])
        if exclude_image_resolution:
            ret_list.append(extra_list["exclude_image_resolution"])
        if include_aesthetic_quality:
            ret_list.append(extra_list["include_aesthetic_quality"])
        if include_composition_style:
            ret_list.append(extra_list["include_composition_style"])
        if exclude_text:
            ret_list.append(extra_list["exclude_text"])
        if specify_depth_field:
            ret_list.append(extra_list["specify_depth_field"])
        if specify_lighting_sources:
            ret_list.append(extra_list["specify_lighting_sources"])
        if do_not_use_ambiguous_language:
            ret_list.append(extra_list["do_not_use_ambiguous_language"])
        if include_nsfw:
            ret_list.append(extra_list["include_nsfw"])
        if only_describe_most_important_elements:
            ret_list.append(extra_list["only_describe_most_important_elements"])

        return ([ret_list, character_name],)


class LS_JoyCaption2:

    CATEGORY = '😺dzNodes/LayerUtility'
    FUNCTION = "joycaption2"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (True,)

    def __init__(self):
        self.NODE_NAME = 'JoyCaption2'
        self.previous_model = None

    @classmethod
    def INPUT_TYPES(self):
        llm_model_list = ["Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2", "unsloth/Meta-Llama-3.1-8B-Instruct"]
        device_list = ['cuda']
        dtype_list = ['nf4','bf16']
        vlm_lora_list = ['text_model', 'none']
        caption_type_list = ["Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney",
                   "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing",
                   "Social Media Post"]
        caption_length_list = ["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261, 10)]

        return {
            "required": {
                "image": ("IMAGE",),
                "llm_model": (llm_model_list,),
                "device": (device_list,),
                "dtype": (dtype_list,),
                "vlm_lora": (vlm_lora_list,),
                "caption_type": (caption_type_list,),
                "caption_length": (caption_length_list,),
                "user_prompt": ("STRING", {"default": "","multiline": False}),
                "max_new_tokens": ("INT", {"default": 300, "min": 8, "max": 4096, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max":1, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0, "max":1, "step": 0.01}),
                "cache_model": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "extra_options": ("JoyCaption2ExtraOption",),
            }
        }

    def joycaption2(self, image, llm_model, device, dtype, vlm_lora, caption_type, caption_length,
                    user_prompt, max_new_tokens, top_p, temperature, cache_model,
                    extra_options=None):

        ret_text = []
        llm_model_path = download_hg_model(llm_model, "LLM")
        if self.previous_model is None:
            model = load_models(llm_model_path, dtype, vlm_lora, device)
        else:
            model = self.previous_model

        extra = []
        character_name = ""
        if extra_options is not None:
            extra, character_name = extra_options

        for img in image:
            img = tensor2pil(img.unsqueeze(0)).convert('RGB')
            # log(f"{self.NODE_NAME}: caption_type={caption_type}, caption_length={caption_length}, extra={extra}, character_name={character_name}, user_prompt={user_prompt}")
            caption = stream_chat([img], caption_type, caption_length,
                                   extra, character_name, user_prompt,
                                   max_new_tokens, top_p, temperature, 1,
                                   model, device)
            log(f"{self.NODE_NAME}: caption={caption[0]}")
            ret_text.append(caption[0])

        if cache_model:
            self.previous_model = model
        else:
            self.previous_model = None
            del model
            clear_memory()

        return (ret_text,)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: JoyCaption2": LS_JoyCaption2,
    "LayerUtility: JoyCaption2ExtraOptions": LS_JoyCaptionExtraOptions
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: JoyCaption2": "LayerUtility: JoyCaption2",
    "LayerUtility: JoyCaption2ExtraOptions": "LayerUtility: JoyCaption2 Extra Options"
}