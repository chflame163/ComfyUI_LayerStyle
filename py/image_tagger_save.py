import os.path
import shutil
from PIL.PngImagePlugin import PngInfo
import datetime
from .imagefunc import *

NODE_NAME = 'ImageTaggerSave'

class LSImageTaggerSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE", ),
                     "tag_text": ("STRING", {"default": "", "forceInput":True}),
                     "custom_path": ("STRING", {"default": ""}),
                     "filename_prefix": ("STRING", {"default": "comfyui"}),
                     "timestamp": (["None", "second", "millisecond"],),
                     "format": (["png", "jpg"],),
                     "quality": ("INT", {"default": 80, "min": 10, "max": 100, "step": 1}),
                     "preview": ("BOOLEAN", {"default": True}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "image_tagger_save"
    OUTPUT_NODE = True
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'

    def image_tagger_save(self, image, tag_text, custom_path, filename_prefix, timestamp, format, quality,
                           preview,
                           prompt=None, extra_pnginfo=None):

        now = datetime.datetime.now()
        custom_path = custom_path.replace("%date", now.strftime("%Y-%m-%d"))
        custom_path = custom_path.replace("%time", now.strftime("%H-%M-%S"))
        filename_prefix = filename_prefix.replace("%date", now.strftime("%Y-%m-%d"))
        filename_prefix = filename_prefix.replace("%time", now.strftime("%H-%M-%S"))
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, image[0].shape[1], image[0].shape[0])
        results = list()
        temp_sub_dir = generate_random_name('_savepreview_', '_temp', 16)
        temp_dir = os.path.join(folder_paths.get_temp_directory(), temp_sub_dir)
        metadata = None
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        if timestamp == "millisecond":
            file = f'{filename}_{now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]}'
        elif timestamp == "second":
            file = f'{filename}_{now.strftime("%Y-%m-%d_%H-%M-%S")}'
        else:
            file = f'{filename}_{counter:08}'

        preview_filename = ""
        if custom_path != "":
            if not os.path.exists(custom_path):
                try:
                    os.makedirs(custom_path)
                except Exception as e:
                    log(f"Error: {NODE_NAME} skipped, because unable to create temporary folder.",
                        message_type='warning')
                    raise FileNotFoundError(f"cannot create custom_path {custom_path}, {e}")

            full_output_folder = os.path.normpath(custom_path)
            # save preview image to temp_dir
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)
            try:
                os.makedirs(temp_dir)
            except Exception as e:
                print(e)
                log(f"Error: {NODE_NAME} skipped, because unable to create temporary folder.",
                    message_type='warning')
            try:
                preview_filename = os.path.join(generate_random_name('saveimage_preview_', '_temp', 16) + '.png')
                img.save(os.path.join(temp_dir, preview_filename))
            except Exception as e:
                print(e)
                log(f"Error: {NODE_NAME} skipped, because unable to create temporary file.", message_type='warning')

            # check if file exists, change filename
            while os.path.isfile(os.path.join(full_output_folder, f"{file}.{format}")):
                counter += 1
                if timestamp == "millisecond":
                    file = f'{filename}_{now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]}_{counter:08}'
                elif timestamp == "second":
                    file = f'{filename}_{now.strftime("%Y-%m-%d_%H-%M-%S")}_{counter:08}'
                else:
                    file = f"{filename}_{counter:08}"

            image_file_name = os.path.join(full_output_folder, f"{file}.{format}")
            tag_file_name = os.path.join(full_output_folder, f"{file}.txt")

            if format == "png":
                img.save(image_file_name, pnginfo=metadata, compress_level= (100 - quality) // 10)
            else:
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(image_file_name, quality=quality)
            with open(tag_file_name, "w", encoding="utf-8") as f:
                f.write(remove_empty_lines(tag_text))
            log(f"{NODE_NAME} -> Saving image to {image_file_name}")

            if preview:
                if custom_path == "":
                    results.append({
                        "filename": f"{file}.{format}",
                        "subfolder": subfolder,
                        "type": self.type
                    })
                else:
                    results.append({
                        "filename": preview_filename,
                        "subfolder": temp_sub_dir,
                        "type": "temp"
                    })

            counter += 1

        return { "ui": { "images": results } }

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageTaggerSave": LSImageTaggerSave
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageTaggerSave": "LayerUtility: Image Tagger Save"
}