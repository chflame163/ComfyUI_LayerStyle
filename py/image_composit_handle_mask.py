from .imagefunc import *

def composite_layer(background_image: Image, layer_image: Image, x_center: int, y_center: int, scale: float = 1.0,
                    rotate: float = 0, aa: int = 1, opacity: int = 100) -> list:
    orig_layer_width, orig_layer_height = layer_image.size
    if aa > 1:
        w, h = layer_image.size
        layer_image = layer_image.resize((w * aa, h * aa), Image.LANCZOS)

    if scale != 1.0:
        w, h = layer_image.size
        layer_image = layer_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    if rotate != 0:
        layer_image = layer_image.rotate(rotate, expand=True, resample=Image.BICUBIC)

    if aa > 1:
        layer_image = layer_image.resize((layer_image.width // aa, layer_image.height // aa), Image.LANCZOS)

    r, g, b, a = layer_image.split()
    alpha = a.copy()
    if 0 <= opacity < 100:
        alpha = alpha.point(lambda i: int(i * opacity * 0.01))
        layer_image = Image.merge("RGBA", (r, g, b, alpha))

    left = int(x_center - layer_image.width / 2)
    top = int(y_center - layer_image.height / 2)

    # composite
    bg = background_image.copy()
    bg.alpha_composite(layer_image, (left, top))
    # draw masks
    whiteimage = Image.new("L", alpha.size, 'white')
    layer_mask = Image.merge("RGBA", (whiteimage, whiteimage, whiteimage, a))
    mask = Image.new("RGBA", bg.size, 'black')
    mask.alpha_composite(layer_mask, (left, top))
    bbox = Image.new("RGBA", bg.size, 'black')
    bbox.alpha_composite(whiteimage.convert("RGBA"), (left, top))

    return [bg.convert("RGB"), mask.convert("L"), bbox.convert("L")]

def sdf_rounded_rect_4corner(px, py, x1, y1, x2, y2, r):
    """
    r = [r_tl, r_tr, r_br, r_bl] 四个角不同半径
    顺序：左上、右上、右下、左下
    """
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    hw = (x2 - x1) * 0.5
    hh = (y2 - y1) * 0.5

    dx = px - cx
    dy = py - cy

    # 按象限选择对应圆角半径
    r_tl, r_tr, r_br, r_bl = r

    # 每个象限对应的半径
    r_corner = np.where(
        (dx < 0) & (dy < 0), r_tl,
        np.where(
            (dx > 0) & (dy < 0), r_tr,
            np.where(
                (dx > 0) & (dy > 0), r_br,
                r_bl
            )
        )
    )

    # 剩余半宽高
    ex = hw - r_corner
    ey = hh - r_corner

    dx2 = np.abs(dx) - ex
    dy2 = np.abs(dy) - ey

    ox = np.maximum(dx2, 0)
    oy = np.maximum(dy2, 0)

    outside = np.hypot(ox, oy)
    inside = np.minimum(np.maximum(dx2, dy2), 0)

    return outside + inside - r_corner


def smoothstep(t):
    return t * t * (3 - 2 * t)


def rounded_rect_gradient_mask_numpy(image, outer_box, inner_box, outer_radius):
    w, h = image.size

    if outer_box[0] <= 0:
        outer_box = [-outer_radius, outer_box[1], outer_box[2], outer_box[3]]
    if outer_box[1] <= 0:
        outer_box = [outer_box[0], -outer_radius, outer_box[2], outer_box[3]]
    if outer_box[2] >= w:
        outer_box = [outer_box[0], outer_box[1], w+outer_radius, outer_box[3]]
    if outer_box[3] >= h:
        outer_box = [outer_box[0], outer_box[1], outer_box[2], h+outer_radius]

    xs = np.arange(w) + 0.5
    ys = np.arange(h) + 0.5
    px, py = np.meshgrid(xs, ys)

    ox1, oy1, ox2, oy2 = outer_box
    ix1, iy1, ix2, iy2 = inner_box

    # ---- 自动禁用圆角边 ----
    inner_radius = outer_radius // 2
    # 四个角默认使用同一半径
    ro = np.array([outer_radius, outer_radius, outer_radius, outer_radius], dtype=np.float32)
    ri = np.array([inner_radius, inner_radius, inner_radius, inner_radius], dtype=np.float32)

    # 左边重叠：左上、左下角 = 0
    if ox1 == ix1:
        ro[0] = ro[3] = 0
        ri[0] = ri[3] = 0

    # 右边重叠：右上、右下角 = 0
    if ox2 == ix2:
        ro[1] = ro[2] = 0
        ri[1] = ri[2] = 0

    # 上边重叠：左上、右上角 = 0
    if oy1 == iy1:
        ro[0] = ro[1] = 0
        ri[0] = ri[1] = 0

    # 下边重叠：左下、右下角 = 0
    if oy2 == iy2:
        ro[3] = ro[2] = 0
        ri[3] = ri[2] = 0

    # ---- 计算 SDF ----
    sd_outer = sdf_rounded_rect_4corner(px, py, ox1, oy1, ox2, oy2, ro)
    sd_inner = sdf_rounded_rect_4corner(px, py, ix1, iy1, ix2, iy2, ri)

    mask = np.zeros((h, w), dtype=np.float32)

    inside_inner = sd_inner < 0
    outside_outer = sd_outer > 0
    transition = ~(inside_inner | outside_outer)

    mask[inside_inner] = 1.0
    mask[outside_outer] = 0.0

    sd_i = sd_inner[transition]
    sd_o = sd_outer[transition]

    t = 1.0 - (sd_i / (sd_i - sd_o + 1e-9))
    t = np.clip(t, 0, 1)

    t = smoothstep(t)

    mask[transition] = t

    return Image.fromarray((mask * 255).astype(np.uint8), mode="L")

class LS_ImageCompositeHandleMask:

    def __init__(self):
        self.NODE_NAME = 'ImageCompositeHandleMask'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        mirror_mode = ['None', 'horizontal', 'vertical']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        handle_detect_list = ['mask_area', 'layer_bbox']
        return {
            "required": {
                "background_image": ("IMAGE",),
                "layer_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "x_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
                "y_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1e4, "step": 0.001}),
                "rotate": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.01}),
                "mirror": (mirror_mode,),
                "anti_aliasing": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "handle_detect": (handle_detect_list,),
                "top_handle": ("FLOAT", {"default": 0.3, "min": 0, "max": 5, "step": 0.01}),
                "bottom_handle": ("FLOAT", {"default": 0.3, "min": 0, "max": 5, "step": 0.01}),
                "left_handle": ("FLOAT", {"default": 0.3, "min": 0, "max": 5, "step": 0.01}),
                "right_handle": ("FLOAT", {"default": 0.3, "min": 0, "max": 5, "step": 0.01}),
                "handle_mask_outradius": ("INT", {"default": 128, "min": 8, "max": 9999, "step": 1}),
                "top_reserve": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "round_to_multiple": (multiple_list,),
            },
            "optional": {
                "layer_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "BOX", "STRING")
    RETURN_NAMES = ("image", "mask", "layer_bbox_mask", "handle_mask", "handle_crop_box", "handle_overrange")
    FUNCTION = 'image_composite_handle_mask'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_composite_handle_mask(self, background_image, layer_image, invert_mask, opacity,
                                    x_percent, y_percent, scale, rotate, mirror, anti_aliasing,
                                    handle_detect, top_handle, bottom_handle, left_handle, right_handle,
                                    handle_mask_outradius,
                                    top_reserve, bottom_reserve, left_reserve, right_reserve,
                                    round_to_multiple, layer_mask=None,):

        ret_images = []
        ret_masks = []
        ret_layer_bbox_masks = []
        ret_handle_masks = []
        handle_overrange = "None"

        b_images = []
        l_images = []
        l_masks = []

        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in layer_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))
        if layer_mask is not None:
            if layer_mask.dim() == 2:
                layer_mask = torch.unsqueeze(layer_mask, 0)
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        max_batch = max(len(b_images), len(l_images), len(l_masks))
        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            layer_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            # preprocess
            _canvas = tensor2pil(background_image).convert('RGBA')
            _layer = tensor2pil(layer_image).convert('RGBA')
            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                log(f"Warning: {self.NODE_NAME} mask mismatch, dropped!", message_type='warning')
            r, g, b, a = _layer.split()
            if 0 <= opacity < 1.0:
                _mask = _mask.point(lambda i: int(i * opacity))
            _layer = Image.merge('RGBA', (r, g, b, _mask))
            # mirror
            if mirror == 'horizontal':
                _layer = _layer.transpose(Image.FLIP_LEFT_RIGHT)
                _mask = _mask.transpose(Image.FLIP_LEFT_RIGHT)
            elif mirror == 'vertical':
                _layer = _layer.transpose(Image.FLIP_TOP_BOTTOM)
                _mask = _mask.transpose(Image.FLIP_TOP_BOTTOM)

            x_center = int(_canvas.width * x_percent / 100)
            y_center = int(_canvas.height * y_percent / 100)
            if anti_aliasing == 0:
                anti_aliasing = 1
            ret_image, ret_mask, bbox_mask = composite_layer(_canvas, _layer, x_center, y_center,
                                                             scale, rotate, anti_aliasing, opacity)

            # clac crop box
            if handle_detect == "mask_area":
                mask_box = mask_area(ret_mask)
            else:
                mask_box = mask_area(bbox_mask)

            x1 = int(mask_box[0]) - left_reserve
            y1 = int(mask_box[1]) - top_reserve
            x2 = int(x1 + mask_box[2]) + right_reserve
            y2 = int(y1 + mask_box[3]) + bottom_reserve
            if x1 < 0:
                x1 = 0
            if x2 > ret_image.width:
                x2 = ret_image.width
            if y1 < 0:
                y1 = 0
            if y2 > ret_image.height:
                y2 = ret_image.height
            mask_box = (x1, y1, x2, y2)

            side_length = ((x2-x1) + (y2-y1)) // 2
            handle_x1 = int(x1 - left_handle * side_length - 1)
            handle_x2 = int(x2 + right_handle * side_length + 1)
            handle_y1 = int(y1 - top_handle * side_length - 1)
            handle_y2 = int(y2 + bottom_handle * side_length + 1)
            handle_width = handle_x2 - handle_x1
            handle_height = handle_y2 - handle_y1

            if round_to_multiple != 'None':
                multiple = int(round_to_multiple)
                handle_width = num_round_up_to_multiple(handle_x2 - handle_x1, multiple)
                handle_height = num_round_up_to_multiple(handle_y2 - handle_y1, multiple)
                handle_x1 = handle_x1 - (handle_width - (handle_x2 - handle_x1)) // 2
                handle_y1 = handle_y1 - (handle_height - (handle_y2 - handle_y1)) // 2
                handle_x2 = handle_x1 + handle_width
                handle_y2 = handle_y1 + handle_height

            if handle_x1 <0:
                handle_x1 = 0
                handle_x2 = num_round_up_to_multiple(handle_x2, multiple)
            if handle_x2 > ret_image.size[0]:
                handle_x1 = handle_x2 - num_round_up_to_multiple(handle_x2 - handle_x1, multiple)
                handle_x2 = ret_image.size[0]
            if handle_y1 <0:
                handle_y1 = 0
                handle_y2 = num_round_up_to_multiple(handle_y2, multiple)
            if handle_y2 > ret_image.size[1]:
                handle_y1 = handle_y2 - num_round_up_to_multiple(handle_y2 - handle_y1, multiple)
                handle_y2 = ret_image.size[1]

            crop_box = (handle_x1, handle_y1, handle_x2, handle_y2)

            # draw handle mask
            handle_mask = rounded_rect_gradient_mask_numpy(ret_mask, crop_box, mask_box, handle_mask_outradius)

            # check handle overrange
            if handle_x1 <= 0 or handle_x2 >= ret_image.size[0] or handle_y1 <= 0 or handle_y2 >= ret_image.size[1]:
                top = ""
                bottom = ""
                left = ""
                right = ""
                if handle_y1 <= 0 :
                    top = "top,"
                if handle_y2 >= ret_image.size[1] :
                    bottom = "bottom,"
                if handle_x1 <= 0 :
                    left = "left,"
                if handle_x2 >= ret_image.size[0] :
                    right = "right"
                handle_overrange = f"{top}{bottom}{left}{right}"
                log(f"{self.NODE_NAME} handle overrange: {handle_overrange}")

            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(ret_mask))
            ret_layer_bbox_masks.append(image2mask(bbox_mask))
            ret_handle_masks.append(image2mask(handle_mask))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),torch.cat(ret_masks, dim=0), torch.cat(ret_layer_bbox_masks, dim=0),
                torch.cat(ret_handle_masks, dim=0), list(crop_box), handle_overrange)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageCompositeHandleMask": LS_ImageCompositeHandleMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageCompositeHandleMask": "LayerUtility: Image Composite Handle Mask",
}

