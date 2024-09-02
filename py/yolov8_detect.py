
import copy
import os.path
from .imagefunc import *

model_path = os.path.join(folder_paths.models_dir, 'yolo')

class YoloV8Detect:

    def __init__(self):
        self.NODE_NAME = 'YoloV8Detect'


    @classmethod
    def INPUT_TYPES(self):
        model_ext = [".pt"]
        FILES_DICT = get_files(model_path, model_ext)
        FILE_LIST = list(FILES_DICT.keys())
        mask_merge = ["all", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        return {
            "required": {
                "image": ("IMAGE", ),
                "yolo_model": (FILE_LIST,),
                "mask_merge": (mask_merge,),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MASK" )
    RETURN_NAMES = ("mask", "yolo_plot_image", "yolo_masks")
    FUNCTION = 'yolo_detect'
    CATEGORY = '😺dzNodes/LayerMask'

    def yolo_detect(self, image,
                          yolo_model, mask_merge
                      ):

        ret_masks = []
        ret_yolo_plot_images = []
        ret_yolo_masks = []

        from  ultralytics import YOLO
        yolo_model = YOLO(os.path.join(model_path, yolo_model))

        for i in image:
            i = torch.unsqueeze(i, 0)
            _image = tensor2pil(i)
            results = yolo_model(_image, retina_masks=True)
            for result in results:
                yolo_plot_image = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                ret_yolo_plot_images.append(pil2tensor(Image.fromarray(yolo_plot_image)))
                # have mask
                if result.masks is not None and len(result.masks) > 0:
                    masks = []
                    masks_data = result.masks.data
                    for index, mask in enumerate(masks_data):
                        _mask = mask.cpu().numpy() * 255
                        _mask = np2pil(_mask).convert("L")
                        ret_yolo_masks.append(image2mask(_mask))
                # no mask, if have box, draw box
                elif result.boxes is not None and len(result.boxes.xyxy) > 0:
                    white_image = Image.new('L', _image.size, "white")
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        _mask = Image.new('L', _image.size, "black")
                        _mask.paste(white_image.crop((x1, y1, x2, y2)), (x1, y1))
                        ret_yolo_masks.append(image2mask(_mask))
                # no mask and box, add a black mask
                else:
                    ret_yolo_masks.append(torch.zeros((1, _image.size[1], _image.size[0]), dtype=torch.float32))
                    # ret_yolo_masks.append(image2mask(Image.new('L', _image.size, "black")))
                    log(f"{NODE_NAME} mask or box not detected.")

                # merge mask
                _mask = ret_yolo_masks[0]
                if mask_merge == "all":
                    for i in range(len(ret_yolo_masks) - 1):
                        _mask = add_mask(_mask, ret_yolo_masks[i + 1])
                else:
                    for i in range(min(len(ret_yolo_masks), int(mask_merge)) - 1):
                        _mask = add_mask(_mask, ret_yolo_masks[i + 1])
                ret_masks.append(_mask)

        log(f"{self.NODE_NAME} Processed {len(ret_masks)} image(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),
                torch.cat(ret_yolo_plot_images, dim=0),
                torch.cat(ret_yolo_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: YoloV8Detect": YoloV8Detect
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: YoloV8Detect": "LayerMask: YoloV8 Detect"
}