
from .imagefunc import *

select_list = ["all", "first", "by_index"]
sort_method_list = ["left_to_right", "top_to_bottom", "big_to_small", "confidence"]


# 规范bbox，保证x1 < x2, y1 < y2, 并返回int
def standardize_bbox(bboxes:list) -> list:
    ret_bboxes = []
    for bbox in bboxes:
        x1 = int(min(bbox[0], bbox[2]))
        y1 = int(min(bbox[1], bbox[3]))
        x2 = int(max(bbox[0], bbox[2]))
        y2 = int(max(bbox[1], bbox[3]))
        ret_bboxes.append([x1, y1, x2, y2])
    return ret_bboxes

def sort_bboxes(bboxes:list, method:str) -> list:
    sorted_bboxes = []
    if method == "left_to_right":
        sorted_bboxes = sorted(bboxes, key=lambda bbox: bbox[0])
    elif method == "top_to_bottom":
        sorted_bboxes = sorted(bboxes, key=lambda bbox: bbox[1])
    elif method == "big_to_small":
        sorted_bboxes = sorted(bboxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), reverse=True)
    else:
        sorted_bboxes = bboxes
    return sorted_bboxes

def select_bboxes(bboxes:list, bbox_select:str, select_index:str) -> list:
    indexs = extract_numbers(select_index)
    if bbox_select == "all":
        return bboxes
    elif bbox_select == "first":
        return [bboxes[0]]
    elif bbox_select == "by_index":
        new_bboxes = []
        for i in indexs:
            try:
                new_bboxes.append(bboxes[i])
            except IndexError:
                log(f"Object detector output by_index: invalid bbox index {i}", message_type='warning')
        return new_bboxes


class LS_BBOXES_JOIN:

    def __init__(self):
        self.NODE_NAME = 'BBoxes Join'

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "bboxes_1": ("BBOXES",),
            },
            "optional": {
                "bboxes_2": ("BBOXES",),
                "bboxes_3": ("BBOXES",),
                "bboxes_4": ("BBOXES",),
            }
        }

    RETURN_TYPES = ("BBOXES",)
    RETURN_NAMES = ("bboxes",)
    FUNCTION = 'bboxes_join'
    CATEGORY = '😺dzNodes/LayerMask'

    def bboxes_join(self, bboxes_1, bboxes_2=None, bboxes_3=None, bboxes_4=None):
        if bboxes_2 is not None:
            bboxes_1.extend(bboxes_2)
        if bboxes_3 is not None:
            bboxes_1.extend(bboxes_3)
        if bboxes_4 is not None:
            bboxes_1.extend(bboxes_4)
        return (bboxes_1,)

class LS_OBJECT_DETECTOR_FL2:

    def __init__(self):
        self.NODE_NAME = 'Object Detector Florence2'

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "prompt": ("STRING", {"default": "subject"}),
                "florence2_model": ("FLORENCE2",),
                "sort_method": (sort_method_list,),
                "bbox_select": (select_list,),
                "select_index": ("STRING", {"default": "0,"},),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BBOXES", "IMAGE",)
    RETURN_NAMES = ("bboxes", "preview",)
    FUNCTION = 'object_detector_fl2'
    CATEGORY = '😺dzNodes/LayerMask'

    def object_detector_fl2(self, image, prompt, florence2_model, sort_method, bbox_select, select_index):

        bboxes = []
        ret_previews = []
        max_new_tokens = 512
        num_beams = 3
        do_sample = False
        fill_mask = False

        model = florence2_model['model']
        processor = florence2_model['processor']

        img = tensor2pil(image[0]).convert("RGB")
        task = 'caption to phrase grounding'
        from .florence2_ultra import  process_image
        results, _ = process_image(model, processor, img, task,
                                   max_new_tokens, num_beams, do_sample,
                                   fill_mask, prompt)

        if isinstance(results, dict):
            results["width"] = img.width
            results["height"] = img.height

        bboxes = self.fbboxes_to_list(results)
        bboxes = sort_bboxes(bboxes, sort_method)
        bboxes = select_bboxes(bboxes, bbox_select, select_index)
        preview = draw_bounding_boxes(img, bboxes, color="random", line_width=-1)
        ret_previews.append(pil2tensor(preview))
        if len(bboxes) == 0:
            log(f"{self.NODE_NAME} no object found", message_type='warning')
        else:
            log(f"{self.NODE_NAME} found {len(bboxes)} object(s)", message_type='info')
        return (standardize_bbox(bboxes), torch.cat(ret_previews, dim=0))

    def fbboxes_to_list(self, F_BBOXES) -> list:
        if isinstance(F_BBOXES, str):
            return None
        ret_bboxes = []
        width = F_BBOXES["width"]
        height = F_BBOXES["height"]
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
                ret_bboxes.append([x1, y1, x2, y2])
        else:
            x1_c = width
            y1_c = height
            x2_c = y2_c = 0
            for polygon in F_BBOXES["polygons"][0]:
                if len(_polygon) < 3:
                    print('Invalid polygon:', _polygon)
                    continue
                x1_c = min(x1_c, int(min(polygon[0::2])))
                x2_c = max(x2_c, int(max(polygon[0::2])))
                y1_c = min(y1_c, int(min(polygon[1::2])))
                y2_c = max(y2_c, int(max(polygon[1::2])))
            ret_bboxes.append(x1_c, y1_c, x2_c, y2_c)
        return ret_bboxes

class LS_OBJECT_DETECTOR_MASK:

    def __init__(self):
        self.NODE_NAME = 'Object Detector MASK'

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "object_mask": ("MASK",),
                "sort_method": (sort_method_list,),
                "bbox_select": (select_list,),
                "select_index": ("STRING", {"default": "0,"},),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BBOXES", "IMAGE",)
    RETURN_NAMES = ("bboxes", "preview",)
    FUNCTION = 'object_detector_mask'
    CATEGORY = '😺dzNodes/LayerMask'

    def object_detector_mask(self, object_mask, sort_method, bbox_select, select_index):

        bboxes = []
        if object_mask.dim() == 2:
            object_mask = torch.unsqueeze(object_mask, 0)

        cv_mask = tensor2cv2(object_mask[0])
        cv_mask = cv2.cvtColor(cv_mask, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(cv_mask, 127, 255, cv2.THRESH_BINARY)
        # invert mask
        # binary = cv2.bitwise_not(binary)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append([x, y, x + w, y + h])
        bboxes = sort_bboxes(bboxes, sort_method)
        bboxes = select_bboxes(bboxes, bbox_select, select_index)
        ret_previews = []
        preview = draw_bounding_boxes(tensor2pil(object_mask[0]).convert("RGB"), bboxes, color="random", line_width=-1)
        ret_previews.append(pil2tensor(preview))

        if len(bboxes) == 0:
            log(f"{self.NODE_NAME} no object found", message_type='warning')
        else:
            log(f"{self.NODE_NAME} found {len(bboxes)} object(s)", message_type='info')

        return (standardize_bbox(bboxes), torch.cat(ret_previews, dim=0))


class LS_OBJECT_DETECTOR_YOLO8:

    def __init__(self):
        self.NODE_NAME = 'Object Detector YOLO8'

    @classmethod
    def INPUT_TYPES(cls):
        model_ext = [".pt"]
        model_path = os.path.join(folder_paths.models_dir, 'yolo')
        FILES_DICT = get_files(model_path, model_ext)
        FILE_LIST = list(FILES_DICT.keys())
        return {
            "required": {
                "image": ("IMAGE", ),
                "yolo_model": (FILE_LIST,),
                "sort_method": (sort_method_list,),
                "bbox_select": (select_list,),
                "select_index": ("STRING", {"default": "0,"},),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BBOXES", "IMAGE",)
    RETURN_NAMES = ("bboxes", "preview",)
    FUNCTION = 'object_detector_yolo8'
    CATEGORY = '😺dzNodes/LayerMask'

    def object_detector_yolo8(self, image, yolo_model, sort_method, bbox_select, select_index):

        from  ultralytics import YOLO
        model_path = os.path.join(folder_paths.models_dir, 'yolo')
        yolo_model = YOLO(os.path.join(model_path, yolo_model))

        bboxes = []
        ret_previews = []

        img = torch.unsqueeze(image[0], 0)
        _image = tensor2pil(img)
        results = yolo_model(_image, retina_masks=True)
        for result in results:
            yolo_plot_image = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

            # no mask, if have box, draw box
            if result.boxes is not None and len(result.boxes.xyxy) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bboxes.append([x1, y1, x2, y2])
        bboxes = sort_bboxes(bboxes, sort_method)
        bboxes = select_bboxes(bboxes, bbox_select, select_index)
        preview = draw_bounding_boxes(_image.convert("RGB"), bboxes, color="random", line_width=-1)
        ret_previews.append(pil2tensor(preview))

        if len(bboxes) == 0:
            log(f"{self.NODE_NAME} no object found", message_type='warning')
        else:
            log(f"{self.NODE_NAME} found {len(bboxes)} object(s)", message_type='info')

        return (standardize_bbox(bboxes), torch.cat(ret_previews, dim=0),)

class LS_OBJECT_DETECTOR_YOLOWORLD:

    def __init__(self):
        self.NODE_NAME = 'Object Detector YOLO-WORLD'
        self.model_path = os.path.join(folder_paths.models_dir, 'yolo-world')
        os.environ['MODEL_CACHE_DIR'] = self.model_path

    @classmethod
    def INPUT_TYPES(cls):
        model_list =['yolo_world/v2-x', 'yolo_world/v2-l', 'yolo_world/v2-m',
                    'yolo_world/v2-s', 'yolo_world/l', 'yolo_world/m',
                    'yolo_world/s']
        return {
            "required": {
                "image": ("IMAGE", ),
                "yolo_world_model": (model_list,),
                "confidence_threshold": ("FLOAT", {"default": 0.05, "min": 0, "max": 1, "step": 0.01}),
                "nms_iou_threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1, "step": 0.01}),
                "prompt": ("STRING", {"default": "subject"}),
                "sort_method": (sort_method_list,),
                "bbox_select": (select_list,),
                "select_index": ("STRING", {"default": "0,"},),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BBOXES", "IMAGE",)
    RETURN_NAMES = ("bboxes", "preview",)
    FUNCTION = 'object_detector_yoloworld'
    CATEGORY = '😺dzNodes/LayerMask'

    def object_detector_yoloworld(self, image, yolo_world_model,
                                  confidence_threshold, nms_iou_threshold, prompt,
                                  sort_method, bbox_select, select_index):
        import supervision as sv

        model=self.load_yolo_world_model(yolo_world_model, prompt)
        infer_outputs = []
        img = (255 * image[0].cpu().numpy()).astype(np.uint8)
        results = model.infer(
            img, confidence=confidence_threshold)
        detections = sv.Detections.from_inference(results)
        detections = detections.with_nms(
            class_agnostic=False,
            threshold=nms_iou_threshold
        )
        infer_outputs.append(detections)
        bboxes = infer_outputs[0].xyxy.tolist()
        bboxes = [[int(value) for value in sublist] for sublist in bboxes]
        bboxes = sort_bboxes(bboxes, sort_method)
        bboxes = select_bboxes(bboxes, bbox_select, select_index)
        ret_previews = []
        preview = draw_bounding_boxes(tensor2pil(image[0]).convert('RGB'), bboxes, color="random", line_width=-1)
        ret_previews.append(pil2tensor(preview))

        if len(bboxes) == 0:
            log(f"{self.NODE_NAME} no object found", message_type='warning')
        else:
            log(f"{self.NODE_NAME} found {len(bboxes)} object(s)", message_type='info')

        return (standardize_bbox(bboxes), torch.cat(ret_previews, dim=0))

    def process_categories(self, categories: str) -> List[str]:
        return [category.strip().lower() for category in categories.split(',')]

    def load_yolo_world_model(self,model_id: str, categories: str) -> List[torch.nn.Module]:
        from inference.models import YOLOWorld as YOLOWorldImpl
        model = YOLOWorldImpl(model_id=model_id)
        categories = self.process_categories(categories)
        model.set_classes(categories)
        return model



class LS_DrawBBoxMask:

    def __init__(self):
        self.NODE_NAME = 'Draw BBOX Mask'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BBOXES",),
                "grow_top": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.01}), # bbox向上扩展，按高度比例
                "grow_bottom": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.01}),
                "grow_left": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.01}),
                "grow_right": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.01}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'draw_bbox_mask'
    CATEGORY = '😺dzNodes/LayerMask'

    def draw_bbox_mask(self, image, bboxes, grow_top, grow_bottom, grow_left, grow_right
                      ):

        ret_masks = []
        for img in image:
            img = tensor2pil(img)
            mask = Image.new("L", img.size, color='black')
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                if grow_top:
                    y1 = int(y1 - h * grow_top)
                if grow_bottom:
                    y2 = int(y2 + h * grow_bottom)
                if grow_left:
                    x1 = int(x1 - w * grow_left)
                if grow_right:
                    x2 = int(x2 + w * grow_right)
                if y1 > y2 or x1 > x2:
                    log(f"{self.NODE_NAME} Invalid bbox after extend: ({x1},{y1},{x2},{y2})", message_type='warning')
                    continue
                draw = ImageDraw.Draw(mask)
                draw.rectangle([x1, y1, x2, y2], fill='white', outline='white', width=0)
                del draw
            ret_masks.append(pil2tensor(mask))

        log(f"{self.NODE_NAME} Processed {len(ret_masks)} mask(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerMask: BBoxJoin": LS_BBOXES_JOIN,
    "LayerMask: DrawBBoxMask": LS_DrawBBoxMask,
    "LayerMask: ObjectDetectorFL2": LS_OBJECT_DETECTOR_FL2,
    "LayerMask: ObjectDetectorMask": LS_OBJECT_DETECTOR_MASK,
    "LayerMask: ObjectDetectorYOLO8": LS_OBJECT_DETECTOR_YOLO8,
    "LayerMask: ObjectDetectorYOLOWorld": LS_OBJECT_DETECTOR_YOLOWORLD
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: BBoxJoin": "LayerMask: BBox Join",
    "LayerMask: DrawBBoxMask": "LayerMask: Draw BBox Mask",
    "LayerMask: ObjectDetectorFL2": "LayerMask: Object Detector Florence2",
    "LayerMask: ObjectDetectorMask": "LayerMask: Object Detector Mask",
    "LayerMask: ObjectDetectorYOLO8": "LayerMask: Object Detector YOLO8",
    "LayerMask: ObjectDetectorYOLOWorld": "LayerMask: Object Detector YOLO World"
}