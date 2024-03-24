from .imagefunc import *
from functools import reduce
import wget
import folder_paths
from .segment_anything_func import *

NODE_NAME = 'PersonMaskUltra'


class PersonMaskUltra:

    def __init__(self):
        # download the model if we need it
        get_a_person_mask_generator_model_path()

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required":
                {
                    "images": ("IMAGE",),
                    "face": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                    "hair": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "body": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "clothes": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "accessories": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "background": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "confidence": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 0.95, "step": 0.01},),
                    "detail_range": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                    "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01}),
                    "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01}),
                    "process_detail": ("BOOLEAN", {"default": True}),
                },
            "optional":
                {
                }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = 'person_mask_ultra'
    CATEGORY = '😺dzNodes/LayerMask'

    def get_mediapipe_image(self, image: Image):
        import mediapipe as mp
        # Convert image to NumPy array
        numpy_image = np.asarray(image)
        image_format = mp.ImageFormat.SRGB
        # Convert BGR to RGB (if necessary)
        if numpy_image.shape[-1] == 4:
            image_format = mp.ImageFormat.SRGBA
        elif numpy_image.shape[-1] == 3:
            image_format = mp.ImageFormat.SRGB
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=image_format, data=numpy_image)

    def person_mask_ultra(self, images, face, hair, body, clothes,
                          accessories, background, confidence,
                          detail_range, black_point, white_point, process_detail):
        import mediapipe as mp
        a_person_mask_generator_model_path = get_a_person_mask_generator_model_path()
        a_person_mask_generator_model_buffer = None
        with open(a_person_mask_generator_model_path, "rb") as f:
            a_person_mask_generator_model_buffer = f.read()
        image_segmenter_base_options = mp.tasks.BaseOptions(model_asset_buffer=a_person_mask_generator_model_buffer)
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=image_segmenter_base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True)
        # Create the image segmenter
        ret_images = []
        ret_masks = []
        with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
            for image in images:
                # image = torch.unsqueeze(image, 0)
                orig_image = tensor2pil(image.unsqueeze(0)).convert('RGB')
                # Convert the Tensor to a PIL image
                # i = 255. * image.cpu().numpy()
                # image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                image_pil = tensor2pil(image.unsqueeze(0)).convert('RGB')
                # create our foreground and background arrays for storing the mask results
                mask_background_array = np.zeros((image_pil.size[0], image_pil.size[1], 4), dtype=np.uint8)
                mask_background_array[:] = (0, 0, 0, 255)
                mask_foreground_array = np.zeros((image_pil.size[0], image_pil.size[1], 4), dtype=np.uint8)
                mask_foreground_array[:] = (255, 255, 255, 255)
                # Retrieve the masks for the segmented image
                media_pipe_image = self.get_mediapipe_image(image=image_pil)
                segmented_masks = segmenter.segment(media_pipe_image)
                masks = []
                if background:
                    masks.append(segmented_masks.confidence_masks[0])
                if hair:
                    masks.append(segmented_masks.confidence_masks[1])
                if body:
                    masks.append(segmented_masks.confidence_masks[2])
                if face:
                    masks.append(segmented_masks.confidence_masks[3])
                if clothes:
                    masks.append(segmented_masks.confidence_masks[4])
                if accessories:
                    masks.append(segmented_masks.confidence_masks[5])
                image_data = media_pipe_image.numpy_view()
                image_shape = image_data.shape
                # convert the image shape from "rgb" to "rgba" aka add the alpha channel
                if image_shape[-1] == 3:
                    image_shape = (image_shape[0], image_shape[1], 4)
                mask_background_array = np.zeros(image_shape, dtype=np.uint8)
                mask_background_array[:] = (0, 0, 0, 255)
                mask_foreground_array = np.zeros(image_shape, dtype=np.uint8)
                mask_foreground_array[:] = (255, 255, 255, 255)
                mask_arrays = []
                if len(masks) == 0:
                    mask_arrays.append(mask_background_array)
                else:
                    for i, mask in enumerate(masks):
                        condition = np.stack((mask.numpy_view(),) * image_shape[-1], axis=-1) > confidence
                        mask_array = np.where(condition, mask_foreground_array, mask_background_array)
                        mask_arrays.append(mask_array)
                # Merge our masks taking the maximum from each
                merged_mask_arrays = reduce(np.maximum, mask_arrays)
                # Create the image
                mask_image = Image.fromarray(merged_mask_arrays)
                # convert PIL image to tensor image
                tensor_mask = mask_image.convert("RGB")
                tensor_mask = np.array(tensor_mask).astype(np.float32) / 255.0
                tensor_mask = torch.from_numpy(tensor_mask)[None,]
                tensor_mask = tensor_mask.squeeze(3)[..., 0]
                _mask = tensor2pil(tensor_mask).convert('L')
                if process_detail:
                    _mask = tensor2pil(mask_edge_detail(pil2tensor(orig_image), pil2tensor(_mask), detail_range, black_point, white_point))
                ret_image = RGB2RGBA(orig_image, _mask)
                ret_images.append(pil2tensor(ret_image))
                ret_masks.append(image2mask(_mask))

            log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
            return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: PersonMaskUltra": PersonMaskUltra
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: PersonMaskUltra": "LayerMask: PersonMaskUltra"
}