from .imagefunc import *

class ImageReelPipeline:
    def __init__(self):
        self.image = None
        self.texts = {}
        self.reel_height = 0
        self.reel_border = 0

Reel = ImageReelPipeline()
class ImageReel:

    def __init__(self):
        self.NODE_NAME = 'ImageReel'

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image1_text": ("STRING", {"multiline": False, "default": "image1"}),
                "image2_text": ("STRING", {"multiline": False, "default": "image2"}),
                "image3_text": ("STRING", {"multiline": False, "default": "image3"}),
                "image4_text": ("STRING", {"multiline": False, "default": "image4"}),
                "reel_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "border": ("INT", {"default": 32, "min": 8, "max": 512}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("Reel",)
    RETURN_NAMES = ("reel",)
    FUNCTION = 'image_reel'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_reel(self, image1, image1_text, image2_text, image3_text, image4_text,
                            reel_height, border,
                            image2=None, image3=None, image4=None,):

        image_list = []
        texts = []
        for img in image1:
            i = self.resize_image_to_height(tensor2pil(img.unsqueeze(0)),reel_height)
            image_list.append(i)
            texts.append([image1_text,i.width])
        if image2 is not None:
            for img in image2:
                i = self.resize_image_to_height(tensor2pil(img.unsqueeze(0)),reel_height)
                image_list.append(i)
                texts.append([image2_text,i.width])
        if image3 is not None:
            for img in image3:
                i = self.resize_image_to_height(tensor2pil(img.unsqueeze(0)),reel_height)
                image_list.append(i)
                texts.append([image3_text,i.width])
        if image4 is not None:
            for img in image4:
                i = self.resize_image_to_height(tensor2pil(img.unsqueeze(0)),reel_height)
                image_list.append(i)
                texts.append([image4_text,i.width])

        reel = ImageReel()
        reel.image = self.draw_reel_image(image_list, border, reel_height)
        reel.texts = texts
        reel.reel_height = reel_height
        reel.reel_border = border
        return (reel,)

    def resize_image_to_height(self, image, target_height) -> Image:
        w = int(target_height / image.height * image.width)
        return image.resize((w, target_height), Image.LANCZOS)

    def draw_reel_image(self, image_list, border, reel_height) -> Image:
        reel_width = 0
        for img in image_list:
            reel_width += img.width + border
        reel_img = Image.new('RGBA', (reel_width, reel_height + border), color=(0, 0, 0, 0))
        #paste images
        w = border // 2
        for img in image_list:
            reel_img.paste(img, (w, border // 2))
            w += img.width + border
        return reel_img


class ImageReelComposit:

    def __init__(self):
        self.NODE_NAME = 'ImageReelComposit'
        (_, self.FONT_DICT) = get_resource_dir()
        self.FONT_LIST = list(self.FONT_DICT.keys())

    @classmethod
    def INPUT_TYPES(self):
        (LUT_DICT, FONT_DICT) = get_resource_dir()
        FONT_LIST = list(FONT_DICT.keys())
        LUT_LIST = list(LUT_DICT.keys())

        color_theme_list = ['light', 'dark']
        return {
            "required": {
                "reel_1": ("Reel",),
                "font_file": (FONT_LIST,),
                "font_size": ("INT", {"default": 40, "min": 4, "max": 1024}),
                "border": ("INT", {"default": 32, "min": 8, "max": 512}),
                "color_theme": (color_theme_list,),
            },
            "optional": {
                "reel_2": ("Reel",),
                "reel_3": ("Reel",),
                "reel_4": ("Reel",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image1",)
    FUNCTION = 'image_reel_composit'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_reel_composit(self, reel_1, font_file, font_size, border, color_theme, reel_2=None, reel_3=None, reel_4=None,):


        ret_images = []

        if color_theme == 'light':
            bg_color = "#E5E5E5"
            text_color = "#121212"
        else:
            bg_color = "#121212"
            text_color = "#E5E5E5"


        font_space = int(font_size * 1.5)
        width = reel_1.image.width
        height = reel_1.image.height + font_space + border
        if reel_2 is not None:
            width = max(width, reel_2.image.width)
            height += reel_2.image.height + font_space + border
        if reel_3 is not None:
            width = max(width, reel_3.image.width)
            height += reel_3.image.height + font_space + border
        if reel_4 is not None:
            width = max(width, reel_4.image.width)
            height += reel_4.image.height + font_space + border

        ret_image = Image.new('RGB', (width, height), color=bg_color)
        paste_y = 0
        reel1_text_image = self.draw_reel_text(reel_1, font_file, font_size, text_color)
        shadow_size = reel_1.image.height // 80
        ret_image = self.paste_drop_shadow(ret_image, reel_1.image, reel1_text_image, ((width - reel_1.image.width) // 2, paste_y),
                                           shadow_size, text_color)

        paste_y += reel_1.image.height + font_space + border
        if reel_2 is not None:
            reel2_text_image = self.draw_reel_text(reel_2, font_file, font_size, text_color)
            shadow_size = reel_2.image.height // 80
            ret_image = self.paste_drop_shadow(ret_image, reel_2.image, reel2_text_image, ((width - reel_2.image.width) // 2, paste_y),
                                               shadow_size, text_color)
            paste_y += reel_2.image.height + font_space + border
        if reel_3 is not None:
            reel3_text_image = self.draw_reel_text(reel_3, font_file, font_size, text_color)
            shadow_size = reel_3.image.height // 80
            ret_image = self.paste_drop_shadow(ret_image, reel_3.image, reel3_text_image,((width - reel_3.image.width) // 2, paste_y),
                                               shadow_size, text_color)
            paste_y += reel_3.image.height + font_space + border
        if reel_4 is not None:
            reel4_text_image = self.draw_reel_text(reel_4, font_file, font_size, text_color)
            shadow_size = reel_4.image.height // 80
            ret_image = self.paste_drop_shadow(ret_image, reel_4.image, reel4_text_image,((width - reel_4.image.width) // 2, paste_y),
                                               shadow_size, text_color)

        ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

    def paste_drop_shadow(self, background_image, image, text_image, box, shadow_size, text_color) -> Image:
        # drop shadow
        _mask = image.split()[3]
        _blured_mask = gaussian_blur(_mask, shadow_size//1.3)
        _blured_mask = adjust_levels(_blured_mask, 0, 255, 0.5, 0, output_white=54).convert('L')
        background_image.paste(Image.new('RGBA', image.size, color="black"), (box[0]+shadow_size, box[1]+shadow_size), mask=_blured_mask)
        background_image.paste(image, box, mask=_mask)
        background_image.paste(Image.new('RGB', text_image.size, color=text_color), (box[0], box[1] + image.height), mask=text_image.split()[3])
        return background_image

    def draw_reel_text(self, reel, font_file, font_size, text_color) -> Image:

        font_path = self.FONT_DICT.get(font_file)
        font = ImageFont.truetype(font_path, font_size)
        texts = reel.texts
        text_image = Image.new('RGBA', (reel.image.width, reel.reel_border + int(font_size * 1.5)), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(text_image)
        x = reel.reel_border
        for t in texts:
            text = t[0]
            width = t[1]
            text_width = font.getbbox(text)[2]
            draw.text(
                xy=(x + width // 2 - text_width//2, reel.reel_border//4),
                text=text,
                fill=text_color,
                font=font,
            )
            x += width + reel.reel_border
        return text_image



NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageReel": ImageReel,
    "LayerUtility: ImageReelComposit": ImageReelComposit
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageReel": "LayerUtility: Image Reel",
    "LayerUtility: ImageReelComposit": "LayerUtility: Image Reel Composit"
}