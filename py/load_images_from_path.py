import os
from PIL import Image, ImageSequence, ImageOps
import torch
import numpy as np
import folder_paths
import node_helpers

class LS_LoadImagesFromPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"placeholder": "c:/images", "images_path": []}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 999999, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = ("images", "masks", "file_name", "frame_count")
    FUNCTION = "ls_load_images"
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'
    OUTPUT_IS_LIST = (True, True, True, False)


    def ls_load_images(self, path: str, image_load_cap: int, select_every_nth: int):
        load_images = []
        load_masks = []
        load_file_names = []
        load_frame_count = 0


        if os.path.isdir(path):
            input_dir = os.path.normpath(path)
            files = [
                os.path.join(input_dir, f)
                for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f))
            ]

            for i in range(len(files)):

                if i % select_every_nth != 0:
                    continue

                image_file = files[i]
                image_path = folder_paths.get_annotated_filepath(image_file)
                img = node_helpers.pillow(Image.open, image_path)
                output_images = []
                output_masks = []
                w, h = None, None

                excluded_formats = ['MPO']

                for i in ImageSequence.Iterator(img):
                    i = node_helpers.pillow(ImageOps.exif_transpose, i)

                    if i.mode == 'I':
                        i = i.point(lambda i: i * (1 / 255))
                    image = i.convert("RGB")

                    if len(output_images) == 0:
                        w = image.size[0]
                        h = image.size[1]

                    if image.size[0] != w or image.size[1] != h:
                        continue

                    image = np.array(image).astype(np.float32) / 255.0
                    image = torch.from_numpy(image)[None,]
                    if 'A' in i.getbands():
                        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                        mask = 1. - torch.from_numpy(mask)
                    elif i.mode == 'P' and 'transparency' in i.info:
                        mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                        mask = 1. - torch.from_numpy(mask)
                    else:
                        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
                    output_images.append(image)
                    output_masks.append(mask.unsqueeze(0))

                if len(output_images) > 1 and img.format not in excluded_formats:
                    output_image = torch.cat(output_images, dim=0)
                    output_mask = torch.cat(output_masks, dim=0)
                else:
                    output_image = output_images[0]
                    output_mask = output_masks[0]

                load_images.append(output_image)
                load_masks.append(output_mask)
                load_file_names.append(os.path.basename(image_file))
                load_frame_count += 1
                if image_load_cap > 0 and load_frame_count >= image_load_cap:
                    break

            return (load_images, load_masks, load_file_names, load_frame_count)

        else:
            raise Exception("directory is not valid: " + directory)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: LoadImagesFromPath": LS_LoadImagesFromPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: LoadImagesFromPath": "LayerUtility: Load Images From Path",
}