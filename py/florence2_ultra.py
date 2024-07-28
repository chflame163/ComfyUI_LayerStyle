import io
from unittest.mock import patch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers.dynamic_module_utils import get_imports
import comfy.model_management
from .imagefunc import *

colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
            'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue']

device = comfy.model_management.get_torch_device()

model_repos = {
    "base": "microsoft/Florence-2-base",
    "base-ft": "microsoft/Florence-2-base-ft",
    "large": "microsoft/Florence-2-large",
    "large-ft": "microsoft/Florence-2-large-ft",
    "DocVQA": "HuggingFaceM4/Florence-2-DocVQA",
    "SD3-Captioner": "gokaygokay/Florence-2-SD3-Captioner",
    "base-PromptGen": "MiaoshouAI/Florence-2-base-PromptGen"
}

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Workaround for FlashAttention"""
    if os.path.basename(filename) != "modeling_florence2.py":
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

def load_model(version):
    florence_path = os.path.join(folder_paths.models_dir, "florence2")
    os.makedirs(florence_path, exist_ok=True)


    model_path = os.path.join(florence_path, version)

    if not os.path.exists(model_path):
        log(f"Downloading Florence2 {version} model...")
        repo_id = model_repos[version]
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=model_path, ignore_patterns=["*.md", "*.txt"])

    try:
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        log(f"Error loading model {version}: {str(e)}")
        log("Attempting to load tokenizer instead of processor...")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            log(f"Error loading model or tokenizer: {str(e)}")

    return (model.to(device), processor)

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    pil = Image.open(buf)
    plt.close()
    return pil

def plot_bbox(image, data):
    fig, ax = plt.subplots()
    fig.set_size_inches(image.width / 100, image.height / 100)
    ax.imshow(image)
    for i, (bbox, label) in enumerate(zip(data['bboxes'], data['labels'])):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        enum_label = f"{i}: {label}"
        plt.text(x1 + 7, y1 + 17, enum_label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    ax.axis('off')
    return fig

def draw_polygons(image, prediction, fill_mask=False):
    output_image = copy.deepcopy(image)
    draw = ImageDraw.Draw(output_image)
    scale = 1
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = color if fill_mask else None
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    return output_image


def convert_to_od_format(data):
    od_results = {
        'bboxes': data.get('bboxes', []),
        'labels': data.get('bboxes_labels', [])
    }
    return od_results


def draw_ocr_bboxes(image, prediction):
    scale = 1
    output_image = copy.deepcopy(image)
    draw = ImageDraw.Draw(output_image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0] + 8, new_box[1] + 2),
                  "{}".format(label),
                  align="right",
                  fill=color)
    return output_image


def run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        early_stopping=False,
        do_sample=do_sample,
        num_beams=num_beams,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer


def process_image(model, processor, image, task_prompt, max_new_tokens, num_beams, do_sample, fill_mask, text_input=None):
    if task_prompt == 'caption':
        task_prompt = '<CAPTION>'
        result = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        return result[task_prompt], None
    elif task_prompt == 'detailed caption':
        task_prompt = '<DETAILED_CAPTION>'
        result = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        return result[task_prompt], None
    elif task_prompt == 'more detailed caption':
        task_prompt = '<MORE_DETAILED_CAPTION>'
        result = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        return result[task_prompt], None
    elif task_prompt == 'object detection':
        task_prompt = '<OD>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        fig = plot_bbox(image, results['<OD>'])
        return results[task_prompt], fig_to_pil(fig)
    elif task_prompt == 'dense region caption':
        task_prompt = '<DENSE_REGION_CAPTION>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        fig = plot_bbox(image, results['<DENSE_REGION_CAPTION>'])
        return results[task_prompt], fig_to_pil(fig)
    elif task_prompt == 'region proposal':
        task_prompt = '<REGION_PROPOSAL>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        fig = plot_bbox(image, results['<REGION_PROPOSAL>'])
        return results[task_prompt], fig_to_pil(fig)
    elif task_prompt == 'caption to phrase grounding':
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input)
        fig = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
        return results[task_prompt], fig_to_pil(fig)
    elif task_prompt == 'referring expression segmentation':
        task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input)
        output_image = draw_polygons(image, results['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask)
        return results[task_prompt], output_image
    elif task_prompt == 'region to segmentation':
        task_prompt = '<REGION_TO_SEGMENTATION>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input)
        output_image = draw_polygons(image, results['<REGION_TO_SEGMENTATION>'], fill_mask)
        return results[task_prompt], output_image
    elif task_prompt == 'open vocabulary detection':
        task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input)
        bbox_results = convert_to_od_format(results['<OPEN_VOCABULARY_DETECTION>'])
        fig = plot_bbox(image, bbox_results)
        return bbox_results, fig_to_pil(fig)
    elif task_prompt == 'region to category':
        task_prompt = '<REGION_TO_CATEGORY>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input)
        return results[task_prompt], None
    elif task_prompt == 'region to description':
        task_prompt = '<REGION_TO_DESCRIPTION>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input)
        return results[task_prompt], None
    elif task_prompt == 'OCR':
        task_prompt = '<OCR>'
        result = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        return result[task_prompt], None
    elif task_prompt == 'OCR with region':
        task_prompt = '<OCR_WITH_REGION>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        output_image = draw_ocr_bboxes(image, results['<OCR_WITH_REGION>'])
        output_results = {'bboxes': results[task_prompt].get('quad_boxes', []),
                          'labels': results[task_prompt].get('labels', [])}
        return output_results, output_image
    else:
        return "", None  # Return empty string and None for unknown task prompts


def remove_angle_bracket_content(text):
    import re
    # 正则表达式匹配 "<>" 包围的内容，包括尖括号本身
    pattern = r'<[^>]*>'
    # 使用 re.sub 替换匹配的内容为空字符串
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def decode_f_bboxes(F_BBOXES):
    if isinstance(F_BBOXES, str):
        return (torch.zeros(1, 512, 512, dtype=torch.float32), F_BBOXES)

    width = F_BBOXES["width"]
    height = F_BBOXES["height"]
    mask = np.zeros((height, width), dtype=np.uint8)

    x1_c = width
    y1_c = height
    x2_c = y2_c = 0
    label = ""
    if "bboxes" in F_BBOXES:
        for idx in range(len(F_BBOXES["bboxes"])):
            bbox = F_BBOXES["bboxes"][idx]

            new_label = F_BBOXES["labels"][idx].removeprefix("</s>")
            if new_label not in label:
                if idx > 0:
                    label = label + ", "
                label = label + new_label

            if len(bbox) == 4:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            elif len(bbox) == 8:
                x1 = int(min(bbox[0::2]))
                x2 = int(max(bbox[0::2]))
                y1 = int(min(bbox[1::2]))
                y2 = int(max(bbox[1::2]))
            else:
                continue

            x1_c = min(x1_c, x1)
            y1_c = min(y1_c, y1)
            x2_c = max(x2_c, x2)
            y2_c = max(y2_c, y2)

            mask[y1:y2, x1:x2] = 1

    else:
        image = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(image)

        x1_c = width
        y1_c = height
        x2_c = y2_c = 0

        for polygon in F_BBOXES["polygons"][0]:
            _polygon = np.array(polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            draw.polygon(_polygon.flatten().tolist(), outline='white', fill='white')

            x1_c = min(x1_c, int(min(polygon[0::2])))
            x2_c = max(x2_c, int(max(polygon[0::2])))
            y1_c = min(y1_c, int(min(polygon[1::2])))
            y2_c = max(y2_c, int(max(polygon[1::2])))

        mask = np.asarray(image)[..., 0].astype(np.float32) / 255

    mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
    # label = remove_angle_bracket_content(label)
    return (mask, label)


class LS_LoadFlorence2Model:
    def __init__(self):
        self.model = None
        self.processor = None
        self.version = None

    @classmethod
    def INPUT_TYPES(s):
        model_list = list(model_repos.keys())
        return {
            "required": {
                "version": (model_list,{"default": model_list[0]}),
            },
        }

    RETURN_TYPES = ("FLORENCE2",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "load"
    CATEGORY = '😺dzNodes/LayerMask'

    def load(self, version):
        if self.version != version:
            self.model, self.processor = load_model(version)
            self.version = version

        return ({'model': self.model, 'processor': self.processor, 'version': self.version, 'device': device},)


class Florence2Ultra:
    def __init__(self):
        self.NODE_NAME = 'Florence2Ultra'

    @classmethod
    def INPUT_TYPES(s):
        segment_task_list = [
            "referring expression segmentation",
            "region to segmentation",
            "open vocabulary detection",
            ]
        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda','cpu']
        return {
            "required": {
                "florence2_model": ("FLORENCE2",),
                "image": ("IMAGE",),
                "task": (segment_task_list,{"default": segment_task_list[0]}),
                "text_input": ("STRING", {"default": "subject"}),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "florence2_ultra"
    CATEGORY = '😺dzNodes/LayerMask'

    def florence2_ultra(self, florence2_model, image, task, text_input,
                        detail_method, detail_erode, detail_dilate,
                        black_point, white_point, process_detail, device, max_megapixels):
        max_new_tokens = 512
        num_beams = 3
        do_sample = False
        fill_mask = False

        ret_images = []
        ret_masks = []

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        model = florence2_model['model']
        processor = florence2_model['processor']

        for i in image:
            img = tensor2pil(i).convert("RGB")

            results, _ = process_image(model, processor, img, task,
                                          max_new_tokens, num_beams, do_sample,
                                          fill_mask, text_input)

            if isinstance(results, dict):
                results["width"] = img.width
                results["height"] = img.height

            _mask, _ = decode_f_bboxes(results)

            if process_detail:
                detail_range = detail_erode + detail_dilate
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(img, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
            else:
                _mask = tensor2pil(_mask)

            ret_image = RGB2RGBA(img, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)


class Florence2Image2Prompt:

    def __init__(self):
        self.NODE_NAME = 'Florence2Image2Prompt'

    @classmethod
    def INPUT_TYPES(s):
        caption_task_list = [
            "caption",
            "detailed caption",
            "more detailed caption",
            "object detection",
            "dense region caption",
            "region proposal",
            "caption to phrase grounding",
            "open vocabulary detection",
            "region to category",
            "region to description",
            "OCR",
            "OCR with region"
            ]
        return {
            "required": {
                "florence2_model": ("FLORENCE2",),
                "image": ("IMAGE",),
                "task": (caption_task_list,{"default": caption_task_list[2]}),
                "text_input": ("STRING", {"default": ""}),
                "max_new_tokens": ("INT", {"default": 1024, "step": 1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "step": 1}),
                "do_sample": ('BOOLEAN', {"default": False}),
                "fill_mask": ('BOOLEAN', {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("text", "preview_image",)
    FUNCTION = "florence2_image2prompt"
    CATEGORY = '😺dzNodes/LayerUtility/Prompt'

    def florence2_image2prompt(self, florence2_model, image, task, text_input,
                               max_new_tokens, num_beams, do_sample, fill_mask):

        model = florence2_model['model']
        processor = florence2_model['processor']

        img = tensor2pil(image[0])
        caption = ""
        results, output_image = process_image(model, processor, img, task, max_new_tokens, num_beams,
                                              do_sample, fill_mask,
                                              text_input)

        if isinstance(results, dict):
            results["width"] = img.width
            results["height"] = img.height

        if output_image == None:
            output_image = image[0].detach().clone().unsqueeze(0)
        else:
            output_image = np.asarray(output_image).astype(np.float32) / 255
            output_image = torch.from_numpy(output_image).unsqueeze(0)

        _, caption = decode_f_bboxes(results)

        return (remove_angle_bracket_content(caption), output_image,)

NODE_CLASS_MAPPINGS = {
    "LayerMask: Florence2Ultra": Florence2Ultra,
    "LayerMask: LoadFlorence2Model": LS_LoadFlorence2Model,
    "LayerUtility: Florence2Image2Prompt": Florence2Image2Prompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: Florence2Ultra": "LayerMask: Florence2 Ultra",
    "LayerMask: LoadFlorence2Model": "LayerMask: Load Florence2 Model",
    "LayerUtility: Florence2Image2Prompt": "LayerUtility: Florence2 Image2Prompt"
}
