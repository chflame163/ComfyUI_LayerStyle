import cv2
import torch
import yaml
import comfy.model_management as mm
from comfy.utils import ProgressBar
from comfy.utils import load_torch_file
from contextlib import nullcontext
from .imagefunc import *

def bboxes2coordinates(bboxes:list) -> list:
    coordinates = []
    for bbox in bboxes:
        coordinates.append(((bbox[0]+bbox[2]) // 2, (bbox[1]+bbox[3]) // 2))
    return coordinates

def load_model(model_path, model_cfg_path, segmentor, dtype, device):
    # import yaml
    from .sam2.modeling.sam2_base import SAM2Base
    from .sam2.modeling.backbones.image_encoder import ImageEncoder
    from .sam2.modeling.backbones.hieradet import Hiera
    from .sam2.modeling.backbones.image_encoder import FpnNeck
    from .sam2.modeling.position_encoding import PositionEmbeddingSine
    from .sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
    from .sam2.modeling.sam.transformer import RoPEAttention
    from .sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock

    from .sam2.sam2_image_predictor import SAM2ImagePredictor
    from .sam2.sam2_video_predictor import SAM2VideoPredictor
    from .sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    # from comfy.utils import load_torch_file

    # Load the YAML configuration
    with open(model_cfg_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract the model configuration
    model_config = config['model']

    # Instantiate the image encoder components
    trunk_config = model_config['image_encoder']['trunk']
    neck_config = model_config['image_encoder']['neck']
    position_encoding_config = neck_config['position_encoding']

    position_encoding = PositionEmbeddingSine(
        num_pos_feats=position_encoding_config['num_pos_feats'],
        normalize=position_encoding_config['normalize'],
        scale=position_encoding_config['scale'],
        temperature=position_encoding_config['temperature']
    )

    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=neck_config['d_model'],
        backbone_channel_list=neck_config['backbone_channel_list'],
        fpn_top_down_levels=neck_config['fpn_top_down_levels'],
        fpn_interp_model=neck_config['fpn_interp_model']
    )

    keys_to_include = ['embed_dim', 'num_heads', 'global_att_blocks', 'window_pos_embed_bkg_spatial_size', 'stages']
    trunk_kwargs = {key: trunk_config[key] for key in keys_to_include if key in trunk_config}
    trunk = Hiera(**trunk_kwargs)

    image_encoder = ImageEncoder(
        scalp=model_config['image_encoder']['scalp'],
        trunk=trunk,
        neck=neck
    )
    # Instantiate the memory attention components
    memory_attention_layer_config = config['model']['memory_attention']['layer']
    self_attention_config = memory_attention_layer_config['self_attention']
    cross_attention_config = memory_attention_layer_config['cross_attention']

    self_attention = RoPEAttention(
        rope_theta=self_attention_config['rope_theta'],
        feat_sizes=self_attention_config['feat_sizes'],
        embedding_dim=self_attention_config['embedding_dim'],
        num_heads=self_attention_config['num_heads'],
        downsample_rate=self_attention_config['downsample_rate'],
        dropout=self_attention_config['dropout']
    )

    cross_attention = RoPEAttention(
        rope_theta=cross_attention_config['rope_theta'],
        feat_sizes=cross_attention_config['feat_sizes'],
        rope_k_repeat=cross_attention_config['rope_k_repeat'],
        embedding_dim=cross_attention_config['embedding_dim'],
        num_heads=cross_attention_config['num_heads'],
        downsample_rate=cross_attention_config['downsample_rate'],
        dropout=cross_attention_config['dropout'],
        kv_in_dim=cross_attention_config['kv_in_dim']
    )

    memory_attention_layer = MemoryAttentionLayer(
        activation=memory_attention_layer_config['activation'],
        dim_feedforward=memory_attention_layer_config['dim_feedforward'],
        dropout=memory_attention_layer_config['dropout'],
        pos_enc_at_attn=memory_attention_layer_config['pos_enc_at_attn'],
        self_attention=self_attention,
        d_model=memory_attention_layer_config['d_model'],
        pos_enc_at_cross_attn_keys=memory_attention_layer_config['pos_enc_at_cross_attn_keys'],
        pos_enc_at_cross_attn_queries=memory_attention_layer_config['pos_enc_at_cross_attn_queries'],
        cross_attention=cross_attention
    )

    memory_attention = MemoryAttention(
        d_model=config['model']['memory_attention']['d_model'],
        pos_enc_at_input=config['model']['memory_attention']['pos_enc_at_input'],
        layer=memory_attention_layer,
        num_layers=config['model']['memory_attention']['num_layers']
    )

    # Instantiate the memory encoder components
    memory_encoder_config = config['model']['memory_encoder']
    position_encoding_mem_enc_config = memory_encoder_config['position_encoding']
    mask_downsampler_config = memory_encoder_config['mask_downsampler']
    fuser_layer_config = memory_encoder_config['fuser']['layer']

    position_encoding_mem_enc = PositionEmbeddingSine(
        num_pos_feats=position_encoding_mem_enc_config['num_pos_feats'],
        normalize=position_encoding_mem_enc_config['normalize'],
        scale=position_encoding_mem_enc_config['scale'],
        temperature=position_encoding_mem_enc_config['temperature']
    )

    mask_downsampler = MaskDownSampler(
        kernel_size=mask_downsampler_config['kernel_size'],
        stride=mask_downsampler_config['stride'],
        padding=mask_downsampler_config['padding']
    )

    fuser_layer = CXBlock(
        dim=fuser_layer_config['dim'],
        kernel_size=fuser_layer_config['kernel_size'],
        padding=fuser_layer_config['padding'],
        layer_scale_init_value=float(fuser_layer_config['layer_scale_init_value'])
    )
    fuser = Fuser(
        num_layers=memory_encoder_config['fuser']['num_layers'],
        layer=fuser_layer
    )

    memory_encoder = MemoryEncoder(
        position_encoding=position_encoding_mem_enc,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
        out_dim=memory_encoder_config['out_dim']
    )

    sam_mask_decoder_extra_args = {
        "dynamic_multimask_via_stability": True,
        "dynamic_multimask_stability_delta": 0.05,
        "dynamic_multimask_stability_thresh": 0.98,
    }

    def initialize_model(model_class, model_config, segmentor, image_encoder, memory_attention, memory_encoder, sam_mask_decoder_extra_args, dtype, device):
        return model_class(
            image_encoder=image_encoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            sam_mask_decoder_extra_args=sam_mask_decoder_extra_args,
            num_maskmem=model_config['num_maskmem'],
            image_size=model_config['image_size'],
            sigmoid_scale_for_mem_enc=model_config['sigmoid_scale_for_mem_enc'],
            sigmoid_bias_for_mem_enc=model_config['sigmoid_bias_for_mem_enc'],
            use_mask_input_as_output_without_sam=model_config['use_mask_input_as_output_without_sam'],
            directly_add_no_mem_embed=model_config['directly_add_no_mem_embed'],
            use_high_res_features_in_sam=model_config['use_high_res_features_in_sam'],
            multimask_output_in_sam=model_config['multimask_output_in_sam'],
            iou_prediction_use_sigmoid=model_config['iou_prediction_use_sigmoid'],
            use_obj_ptrs_in_encoder=model_config['use_obj_ptrs_in_encoder'],
            add_tpos_enc_to_obj_ptrs=model_config['add_tpos_enc_to_obj_ptrs'],
            only_obj_ptrs_in_the_past_for_eval=model_config['only_obj_ptrs_in_the_past_for_eval'],
            pred_obj_scores=model_config['pred_obj_scores'],
            pred_obj_scores_mlp=model_config['pred_obj_scores_mlp'],
            fixed_no_obj_ptr=model_config['fixed_no_obj_ptr'],
            multimask_output_for_tracking=model_config['multimask_output_for_tracking'],
            use_multimask_token_for_obj_ptr=model_config['use_multimask_token_for_obj_ptr'],
            compile_image_encoder=model_config['compile_image_encoder'],
            multimask_min_pt_num=model_config['multimask_min_pt_num'],
            multimask_max_pt_num=model_config['multimask_max_pt_num'],
            use_mlp_for_obj_ptr_proj=model_config['use_mlp_for_obj_ptr_proj'],
            proj_tpos_enc_in_obj_ptrs=model_config['proj_tpos_enc_in_obj_ptrs'],
            no_obj_embed_spatial=model_config['no_obj_embed_spatial'],
            use_signed_tpos_enc_to_obj_ptrs=model_config['use_signed_tpos_enc_to_obj_ptrs'],
            binarize_mask_from_pts_for_mem_enc=True if segmentor == 'video' else False,
        ).to(dtype).to(device).eval()

    # Load the state dictionary
    sd = load_torch_file(model_path)

    # Initialize model based on segmentor type
    if segmentor == 'single_image':
        model_class = SAM2Base
        model = initialize_model(model_class, model_config, segmentor, image_encoder, memory_attention, memory_encoder, sam_mask_decoder_extra_args, dtype, device)
        model.load_state_dict(sd)
        model = SAM2ImagePredictor(model)
    elif segmentor == 'video':
        model_class = SAM2VideoPredictor
        model = initialize_model(model_class, model_config, segmentor, image_encoder, memory_attention, memory_encoder, sam_mask_decoder_extra_args, dtype, device)
        model.load_state_dict(sd)
    elif segmentor == 'automaskgenerator':
        model_class = SAM2Base
        model = initialize_model(model_class, model_config, segmentor, image_encoder, memory_attention, memory_encoder, sam_mask_decoder_extra_args, dtype, device)
        model.load_state_dict(sd)
        model = SAM2AutomaticMaskGenerator(model)
    else:
        raise ValueError(f"Segmentor {segmentor} not supported")

    return model


class LS_SAM2_ULTRA:

    def __init__(self):
        self.NODE_NAME = 'SAM2 Ultra'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        sam2_model_list = ['sam2_hiera_base_plus.safetensors',
                           'sam2_hiera_large.safetensors',
                           'sam2_hiera_small.safetensors',
                           'sam2_hiera_tiny.safetensors',
                           'sam2.1_hiera_base_plus.safetensors',
                           'sam2.1_hiera_large.safetensors',
                           'sam2.1_hiera_small.safetensors',
                           'sam2.1_hiera_tiny.safetensors',
                           ]
        model_precision_list = [ 'fp16','bf16','fp32']
        select_list = ["all", "first", "by_index"]
        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda','cpu']
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BBOXES",),
                "sam2_model": (sam2_model_list,),
                "precision": (model_precision_list,),
                "bbox_select": (select_list,),
                "select_index": ("STRING", {"default": "0,"},),
                "cache_model": ("BOOLEAN", {"default": False}),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 4, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = 'sam2_ultra'
    CATEGORY = '😺dzNodes/LayerMask'

    def sam2_ultra(self, image, bboxes, sam2_model, precision,
                   bbox_select, select_index, cache_model,
                   detail_method, detail_erode, detail_dilate, black_point, white_point,
                   process_detail, device, max_megapixels,
                   ):

        ret_images = []
        ret_masks = []

        # load model
        sam2_path = os.path.join(folder_paths.models_dir, "sam2")
        if precision != 'fp32' and "2.1" in sam2_model:
            base_name, extension = sam2_model.rsplit('.', 1)
            sam2_model = f"{base_name}-fp16.{extension}"
        model_path = os.path.join(sam2_path, sam2_model)

        if device == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        # device = {"cuda": torch.device("cuda"), "cpu": torch.device("cpu")}[device]
        segmentor = 'single_image'
        if not os.path.exists(model_path):
            log(f"{self.NODE_NAME}: Downloading SAM2 model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/sam2-safetensors",
                            allow_patterns=[f"*{sam2_model}*"],
                            local_dir=sam2_path,
                            local_dir_use_symlinks=False)

        model_mapping = {
            "2.0": {
                "base": "sam2_hiera_b+.yaml",
                "large": "sam2_hiera_l.yaml",
                "small": "sam2_hiera_s.yaml",
                "tiny": "sam2_hiera_t.yaml"
            },
            "2.1": {
                "base": "sam2.1_hiera_b+.yaml",
                "large": "sam2.1_hiera_l.yaml",
                "small": "sam2.1_hiera_s.yaml",
                "tiny": "sam2.1_hiera_t.yaml"
            }
        }
        version = "2.1" if "2.1" in sam2_model else "2.0"

        model_cfg_path = next(
            (os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2", "sam2_configs", cfg)
             for key, cfg in model_mapping[version].items() if key in sam2_model),
            None
        )
        log(f"{self.NODE_NAME}: Using model config: {model_cfg_path}")

        model = load_model(model_path, model_cfg_path, segmentor, dtype, device)

        offload_device = mm.unet_offload_device()

        # B, H, W, C = image.shape
        indexs = extract_numbers(select_index)

        # Handle possible bboxes
        if len(bboxes) == 0:
            log(f"{self.NODE_NAME} skipped, because bboxes is empty.", message_type='error')
            return (image, None)
        else:
            boxes_np_batch = []
            for bbox_list in bboxes:
                boxes_np = []
                for bbox in bbox_list:
                    boxes_np.append(bbox)
                boxes_np = np.array(boxes_np)
                boxes_np_batch.append(boxes_np)
            if bbox_select == "all":
                final_box = np.array(boxes_np_batch)
            elif bbox_select == "by_index":
                final_box = []
                try:
                    for i in indexs:
                        final_box.append(boxes_np_batch[i])
                except IndexError:
                    log(f"{self.NODE_NAME} invalid bbox index {i}", message_type='warning')
            else:
                final_box = np.array(boxes_np_batch[0])
            # final_labels = None

        mask_list = []
        try:
            model.to(device)
        except:
            model.model.to(device)

        autocast_condition = not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():

            image_np = (image.contiguous() * 255).byte().numpy()
            comfy_pbar = ProgressBar(len(image_np))
            tqdm_pbar = tqdm(total=len(image_np), desc="Processing Images")
            for i in range(len(image_np)):
                model.set_image(image_np[i])
                # if len(image_np) > 1:
                #     input_box = final_box[i]
                input_box = final_box

                out_masks, scores, logits = model.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=True,
                    mask_input=None,
                )

                if out_masks.ndim == 3:
                    sorted_ind = np.argsort(scores)[::-1]
                    out_masks = out_masks[sorted_ind][0]  # choose only the best result for now
                    # scores = scores[sorted_ind]
                    # logits = logits[sorted_ind]
                    mask_list.append(np.expand_dims(out_masks, axis=0))
                else:
                    _, _, H, W = out_masks.shape
                    # Combine masks for all object IDs in the frame
                    combined_mask = np.zeros((H, W), dtype=bool)
                    for out_mask in out_masks:
                        combined_mask = np.logical_or(combined_mask, out_mask)
                    combined_mask = combined_mask.astype(np.uint8)
                    mask_list.append(combined_mask)
                comfy_pbar.update(1)
                tqdm_pbar.update(1)

        if cache_model:
            try:
                model.to(offload_device)
            except:
                try:
                    model.model.to(offload_device)
                except:
                    pass
        else:
            del model
            clear_memory()

        out_list = []
        for mask in mask_list:
            mask_tensor = torch.from_numpy(mask)
            mask_tensor = mask_tensor.permute(1, 2, 0)
            mask_tensor = mask_tensor[:, :, 0]
            out_list.append(mask_tensor)
        mask_tensor = torch.stack(out_list, dim=0).cpu().float()
        _mask = mask_tensor.squeeze()

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False
        orig_image = tensor2pil(image[0])
        detail_range = detail_erode + detail_dilate
        if process_detail:
            if detail_method == 'GuidedFilter':
                _mask = guided_filter_alpha(pil2tensor(orig_image), _mask, detail_range // 6 + 1)
                _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
            elif detail_method == 'PyMatting':
                _mask = tensor2pil(mask_edge_detail(pil2tensor(orig_image), _mask, detail_range // 8 + 1, black_point, white_point))
            else:
                _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                _mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device,
                                          max_megapixels=max_megapixels)
                _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
        else:
            _mask = tensor2pil(_mask)

        ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
        ret_images.append(pil2tensor(ret_image))
        ret_masks.append(image2mask(_mask))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0))

# 在mask范围内随机生成指定数量的点
def poisson_disk_sampling(mask:Image, radius:float=32, num_points:int=16) -> list:
    """
    使用泊松盘采样在掩码的白色区域内生成点，确保每个点之间至少为radius像素。

    参数:
    - mask: PIL.Image对象，将转换为numpy数组，二值化的掩码图像，白色区域为1，黑色区域为0
    - radius: float，点之间的最小距离
    - num_points: int，期望生成的点的数量

    返回:
    - points: list of (x, y)元组，生成的点的坐标
    """

    gray_mask = np.asarray(mask.convert('L'))
    # gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_mask, 127, 1, cv2.THRESH_BINARY)
    # binary_mask = binary_mask.astype(np.uint8)
    # 计算距离变换
    distance = cv2.distanceTransform(binary_mask, distanceType=cv2.DIST_L2, maskSize=5)

    # 使用泊松盘采样算法
    from skimage.feature import peak_local_max
    coordinates = peak_local_max(distance, min_distance=radius, num_peaks=num_points, exclude_border=True)

    # 将坐标转换为列表形式
    points = [tuple(pt[::-1]) for pt in coordinates]  # (x, y)

    return points


class LS_SAM2_VIDEO_ULTRA:


    def __init__(self):
        self.NODE_NAME = 'SAM2 Video Ultra'

    @classmethod
    def INPUT_TYPES(cls):
        sam2_model_list = ['sam2_hiera_base_plus.safetensors',
                           'sam2_hiera_large.safetensors',
                           'sam2_hiera_small.safetensors',
                           'sam2_hiera_tiny.safetensors',
                           'sam2.1_hiera_base_plus.safetensors',
                           'sam2.1_hiera_large.safetensors',
                           'sam2.1_hiera_small.safetensors',
                           'sam2.1_hiera_tiny.safetensors',
                           ]
        model_precision_list = ['fp16','bf16']
        method_list = ['VITMatte']
        device_list = ['cuda']
        return {
            "required": {
                "image": ("IMAGE",),
                "sam2_model": (sam2_model_list,),
                "precision": (model_precision_list,),
                "cache_model": ("BOOLEAN", {"default": False}),
                "individual_objects": ("BOOLEAN", {"default": False}),
                "mask_preview_color": ("STRING", {"default": "#FF0080"},),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 4, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10, "step": 0.1}),
            },
            "optional": {
                "bboxes": ("BBOXES",),
                "first_frame_mask": ("MASK",),
                "pre_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK","IMAGE")
    RETURN_NAMES = ("mask","preview")
    FUNCTION = 'sam2_video_ultra'
    CATEGORY = '😺dzNodes/LayerMask'

    def sam2_video_ultra(self, image, sam2_model, precision,
                         cache_model, individual_objects, mask_preview_color,
                         detail_method, detail_erode, detail_dilate, black_point, white_point,
                         process_detail, device, max_megapixels,
                         bboxes = None, first_frame_mask=None, pre_mask=None
                         ):

        if first_frame_mask is None:
            if bboxes is None:
                log(f"{self.NODE_NAME} skipped, first_frame_mask or bboxes must have input.", message_type='error')
                return (image, None)
            elif len(bboxes) == 0:
                log(f"{self.NODE_NAME} skipped, because first_frame_mask is none and bboxes is empty.", message_type='error')
                return (image, None)

        # load model
        sam2_path = os.path.join(folder_paths.models_dir, "sam2")
        if precision != 'fp32' and "2.1" in sam2_model:
            base_name, extension = sam2_model.rsplit('.', 1)
            sam2_model = f"{base_name}-fp16.{extension}"
        model_path = os.path.join(sam2_path, sam2_model)

        if device == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        if not os.path.exists(model_path):
            log(f"{self.NODE_NAME}: Downloading SAM2 model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/sam2-safetensors",
                              allow_patterns=[f"*{sam2_model}*"],
                              local_dir=sam2_path,
                              local_dir_use_symlinks=False)

        model_mapping = {
            "2.0": {
                "base": "sam2_hiera_b+.yaml",
                "large": "sam2_hiera_l.yaml",
                "small": "sam2_hiera_s.yaml",
                "tiny": "sam2_hiera_t.yaml"
            },
            "2.1": {
                "base": "sam2.1_hiera_b+.yaml",
                "large": "sam2.1_hiera_l.yaml",
                "small": "sam2.1_hiera_s.yaml",
                "tiny": "sam2.1_hiera_t.yaml"
            }
        }
        version = "2.1" if "2.1" in sam2_model else "2.0"

        model_cfg_path = next(
            (os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2", "sam2_configs", cfg)
             for key, cfg in model_mapping[version].items() if key in sam2_model),
            None
        )
        log(f"{self.NODE_NAME}: Using model config: {model_cfg_path}")

        offload_device = mm.unet_offload_device()
        B, H, W, C = image.shape

        if pre_mask is not None:
            input_mask = pre_mask.clone().unsqueeze(1)
            input_mask = F.interpolate(input_mask, size=(256, 256), mode="bilinear")
            input_mask = input_mask.squeeze(1)

        autocast_condition = not mm.is_device_mps(device)

        # init video model
        v_model = load_model(model_path, model_cfg_path, 'video', dtype, device)
        model_input_image_size = v_model.image_size
        from comfy.utils import common_upscale
        resized_image = common_upscale(image.movedim(-1,1), model_input_image_size, model_input_image_size, "bilinear", "disabled").movedim(1,-1)
        try:
            v_model.to(device)
        except:
            v_model.model.to(device)
        s_model = None
        if first_frame_mask is None:
            # load single_image_model
            s_model = load_model(model_path, model_cfg_path, 'single_image', dtype, device)

            # gen first frame mask
            with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
                f_mask = []
                boxes_np_batch = []
                for bbox_list in bboxes:
                    boxes_np = []
                    for bbox in bbox_list:
                        boxes_np.append(bbox)
                    boxes_np = np.array(boxes_np)
                    boxes_np_batch.append(boxes_np)
                final_box = np.array(boxes_np_batch)
                final_labels = None

                image_np = (image.contiguous() * 255).byte().numpy()
                i = 0
                s_model.set_image(image_np[i])
                input_box = final_box

                out_masks, scores, logits = s_model.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=True,
                    # mask_input=None,
                    mask_input=input_mask[0].unsqueeze(0) if pre_mask is not None else None,
                )

                if out_masks.ndim == 3:
                    sorted_ind = np.argsort(scores)[::-1]
                    out_masks = out_masks[sorted_ind][0]  # choose only the best result for now
                    # scores = scores[sorted_ind]
                    # logits = logits[sorted_ind]
                    f_mask.append(np.expand_dims(out_masks, axis=0))
                else:
                    _, _, H, W = out_masks.shape
                    # Combine masks for all object IDs in the frame
                    combined_mask = np.zeros((H, W), dtype=bool)
                    for out_mask in out_masks:
                        combined_mask = np.logical_or(combined_mask, out_mask)
                    combined_mask = combined_mask.astype(np.uint8)
                    f_mask.append(combined_mask)

                out_list = []
                for mask in f_mask:
                    mask_tensor = torch.from_numpy(mask)
                    mask_tensor = mask_tensor.permute(1, 2, 0)
                    mask_tensor = mask_tensor[:, :, 0]
                    out_list.append(mask_tensor)
                mask_tensor = torch.stack(out_list, dim=0).cpu().float()
                f_mask = tensor2pil(mask_tensor.squeeze()).convert("L")
        else:
            if first_frame_mask.dim() == 2:
                first_frame_mask = torch.unsqueeze(first_frame_mask, 0)
            f_mask = tensor2pil(first_frame_mask[0])
        coords = poisson_disk_sampling(f_mask, radius=32, num_points=16)

        # gen video mask
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            if not individual_objects:
                positive_point_coords = np.atleast_2d(np.array(coords))
            else:
                positive_point_coords = np.array([np.atleast_2d(coord) for coord in coords])

            if not individual_objects:
                positive_point_labels = np.ones(len(positive_point_coords))
            else:
                positive_labels = []
                for point in positive_point_coords:
                    positive_labels.append(np.array([1]))  # 1)
                positive_point_labels = np.stack(positive_labels, axis=0)

            final_coords = positive_point_coords
            final_labels = positive_point_labels

            mask_list = []
            if hasattr(self, 'inference_state'):
                v_model.reset_state(self.inference_state)
            self.inference_state = v_model.init_state(resized_image.permute(0, 3, 1, 2).contiguous(), H, W, device=device)

            if individual_objects:
                for i, (coord, label) in enumerate(zip(final_coords, final_labels)):
                    _, out_obj_ids, out_mask_logits = v_model.add_new_points(
                        inference_state=self.inference_state,
                        frame_idx=0,
                        obj_id=i,
                        points=final_coords[i],
                        labels=final_labels[i],
                    )
            else:
                _, out_obj_ids, out_mask_logits = v_model.add_new_points(
                    inference_state=self.inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=final_coords,
                    labels=final_labels,
                )

            pbar = ProgressBar(B)
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in v_model.propagate_in_video(self.inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                pbar.update(1)
                if individual_objects:
                    _, _, H, W = out_mask_logits.shape
                    # Combine masks for all object IDs in the frame
                    combined_mask = np.zeros((H, W), dtype=np.uint8)
                    for i, out_obj_id in enumerate(out_obj_ids):
                        out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                        combined_mask = np.logical_or(combined_mask, out_mask)
                    video_segments[out_frame_idx] = combined_mask

            if individual_objects:
                for frame_idx, combined_mask in video_segments.items():
                    mask_list.append(combined_mask)
            else:
                for frame_idx, obj_masks in video_segments.items():
                    for out_obj_id, out_mask in obj_masks.items():
                        mask_list.append(out_mask)

        if cache_model:
            try:
                v_model.to(offload_device)
                s_model.to(offload_device)
            except:
                try:
                    v_model.model.to(offload_device)
                    s_model.model.to(offload_device)
                except:
                    pass
        else:
            del v_model
            del s_model
            clear_memory()

        out_list = []
        for mask in mask_list:
            mask_tensor = torch.from_numpy(mask)
            mask_tensor = mask_tensor.permute(1, 2, 0)
            mask_tensor = mask_tensor[:, :, 0]
            out_list.append(mask_tensor)
        out_list = torch.stack(out_list, dim=0).cpu().float()

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        detail_range = detail_erode + detail_dilate
        ret_previews = []
        ret_masks = []
        from tqdm import tqdm
        comfy_pbar = ProgressBar(len(image))
        tqdm_pbar = tqdm(total=len(image), desc="processing masks")
        for index, img in tqdm(enumerate(image)):
            orig_image = tensor2pil(img)
            _mask = out_list[index].unsqueeze(0)
            _mask = tensor2pil(_mask).resize((orig_image.size), Image.BILINEAR)
            if process_detail:
                _trimap = generate_VITMatte_trimap(pil2tensor(_mask), detail_erode, detail_dilate)
                _mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device,
                                          max_megapixels=max_megapixels)
                _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))

            color_image = Image.new("RGB", orig_image.size,color=mask_preview_color)
            color_image = chop_image_v2(orig_image, color_image, "normal", 50)
            color_image.paste(orig_image, mask=_mask)
            ret_previews.append(pil2tensor(color_image))
            ret_masks.append(image2mask(_mask))
            comfy_pbar.update(1)
            tqdm_pbar.update(1)

        log(f"{self.NODE_NAME} Processed {len(ret_masks)} frame(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0), torch.cat(ret_previews, dim=0))


NODE_CLASS_MAPPINGS = {
    "LayerMask: SAM2Ultra": LS_SAM2_ULTRA,
    "LayerMask: SAM2VideoUltra": LS_SAM2_VIDEO_ULTRA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: SAM2Ultra": "LayerMask: SAM2 Ultra",
    "LayerMask: SAM2VideoUltra": "LayerMask: SAM2 Video Ultra"
}