import os.path
from PIL.PngImagePlugin import PngInfo
import datetime
from .imagefunc import *

NODE_NAME = 'SaveImagePlus'

class SaveImagePlus:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                     "custom_path": ("STRING", {"default": ""}),
                     "filename_prefix": ("STRING", {"default": "comfyui"}),
                     "timestamp": (["None", "second", "millisecond"],),
                     "format": (["png", "jpg"],),
                     "quality": ("INT", {"default": 100, "min": 10, "max": 100, "step": 1}),
                     "meta_data": ("BOOLEAN", {"default": False}),
                     "blind_watermark": ("STRING", {"default": ""}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_image_plus"
    OUTPUT_NODE = True
    CATEGORY = '😺dzNodes/LayerUtility'

    def save_image_plus(self, images, custom_path, filename_prefix, timestamp, format, quality,
                                  meta_data, blind_watermark, prompt=None, extra_pnginfo=None):


        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if blind_watermark != "":
                img_mode = img.mode
                wm_size = watermark_image_size(img)
                import qrcode
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_H,
                    box_size=20,
                    border=1,
                )
                qr.add_data(blind_watermark.encode('utf-8'))
                qr.make(fit=True)
                qr_image = qr.make_image(fill_color="black", back_color="white")
                qr_image = qr_image.resize((wm_size, wm_size), Image.BICUBIC).convert("L")

                y, u, v, _ = image_channel_split(img, mode='YCbCr')
                _u = add_invisibal_watermark(u, qr_image)
                wm_img = image_channel_merge((y, _u, v), mode='YCbCr')

                if img.mode == "RGBA":
                    img = RGB2RGBA(wm_img, img.split()[-1])
                else:
                    img = wm_img.convert(img_mode)

            metadata = None
            if meta_data:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            if timestamp == "millisecond":
                file = f'{filename}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]}.{format}'
            elif timestamp == "second":
                file = f'{filename}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.{format}'
            else:
                file = f'{filename}_{counter:05}.{format}'

            if not os.path.exists(custom_path):
                if custom_path != "":
                    raise FileNotFoundError("custom_path is not a valid path")
            else:
                full_output_folder = os.path.normpath(custom_path)

            while os.path.isfile(os.path.join(full_output_folder, file)):
                counter += 1
                if timestamp == "millisecond":
                    file = f'{filename}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]}_{counter:05}.{format}'
                elif timestamp == "second":
                    file = f'{filename}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{counter:05}.{format}'
                else:
                    file = f"{filename}_{counter:05}.{format}"

            if format == "png":
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level= (100 - quality) // 10)
            else:
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(os.path.join(full_output_folder, file), quality=quality)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

NODE_CLASS_MAPPINGS = {
    "LayerUtility: SaveImagePlus": SaveImagePlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: SaveImagePlus": "LayerUtility: SaveImage Plus"
}