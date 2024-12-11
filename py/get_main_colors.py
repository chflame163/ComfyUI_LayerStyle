import torch
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import colorsys
from .imagefunc import AnyType, log, tensor2pil, pil2tensor, load_custom_size, gaussian_blur
from .imagefunc import RGB_to_Hex

any = AnyType("*")


class LS_GetMainColorsV2:

    def __init__(self):
        self.NODE_NAME = 'Get Main Colors V2'

    @classmethod
    def INPUT_TYPES(self):
        size_list = ['custom']
        size_list.extend(load_custom_size())
        k_means_algorithm_list = ["lloyd", "elkan"]
        return {
            "required": {
                "image": ("IMAGE",),
                "k_means_algorithm": (k_means_algorithm_list,),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING","STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("preview_image", "color_1", "color_2", "color_3", "color_4", "color_5",)
    FUNCTION = 'get_main_colors_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def get_main_colors_v2(self, image, k_means_algorithm):
        ret_images = []
        grid_width = 512
        grid_height = 64  # Reduced height to fit 10 colors

        for i in range(len(image)):
            pil_img = tensor2pil(torch.unsqueeze(image[i], 0)).convert("RGB")
            blured_image = gaussian_blur(pil_img, (pil_img.width + pil_img.height) // 400)

            accuracy = 60
            num_colors = 5  # Increased to 5 colors
            num_iterations = int(512 * (accuracy / 100))
            original_colors, color_percentages = self.interrogate_colors(
                pil2tensor(blured_image), num_colors=num_colors, algorithm=k_means_algorithm, mix_iter=num_iterations,
                random_state=0)

            main_colors = self.ndarrays_to_colorhex(original_colors)

            # Sort colors by percentage
            sorted_colors = sorted(zip(main_colors, color_percentages), key=lambda x: x[1], reverse=True)
            print(f"sorted_colors={sorted_colors},type={type(sorted_colors)}")
            # Create color info string with HSB values
            color_info = "\n".join([
                f"RGB {color[1:]}   HSB {self.rgb_to_hsb(color)[0]:03.0f} {self.rgb_to_hsb(color)[1]:03.0f} {self.rgb_to_hsb(color)[2]:03.0f}   占比 {percentage:.2f}%"
                for color, percentage in sorted_colors
            ])

            # draw colors image
            ret_image = Image.new('RGB', size=(grid_width, grid_height * len(main_colors)), color="white")
            draw = ImageDraw.Draw(ret_image)

            # Use default font with size 20
            font = ImageFont.load_default().font_variant(size=20)

            for j, (color, percentage) in enumerate(sorted_colors):
                x1 = 0
                y1 = grid_height * j
                draw.rectangle((x1, y1, x1 + grid_width, y1 + grid_height), fill=color, outline=color)

                # Calculate contrast color
                contrast_color = self.get_contrast_color(color)

                # Add text with contrast color and HSB values
                h, s, b = self.rgb_to_hsb(color)
                text = f"RGB {color[1:]}   HSB {h:03.0f} {s:03.0f} {b:03.0f}   {percentage:.2f}%"
                # 使用 font.getbbox() 来获取文本的边界框
                bbox = font.getbbox(text)
                text_height = bbox[3] - bbox[1]

                # 计算文本的垂直位置，使其在色块中垂直居中
                text_x = 10  # 固定左边距为10像素
                text_y = y1 + (grid_height - text_height) // 2

                # 绘制文本
                draw.text((text_x, text_y), text, fill=contrast_color, font=font)

            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), sorted_colors[0][0], sorted_colors[1][0], sorted_colors[2][0], sorted_colors[3][0], sorted_colors[4][0],)

    def ndarrays_to_colorhex(self, colors: list) -> list:
        return [RGB_to_Hex((int(color[0]), int(color[1]), int(color[2]))) for color in colors]

    def interrogate_colors(self, image: torch.Tensor, num_colors: int, algorithm: str, mix_iter: int,
                           random_state: int) -> tuple:
        from sklearn.cluster import KMeans
        pixels = image.view(-1, image.shape[-1]).numpy()
        kmeans = KMeans(
            n_clusters=num_colors,
            algorithm=algorithm,
            max_iter=mix_iter,
            random_state=random_state,
        ).fit(pixels)

        colors = kmeans.cluster_centers_ * 255

        # Count pixels in each cluster
        labels = kmeans.labels_
        label_counts = Counter(labels)
        total_pixels = len(labels)

        # Calculate percentages
        color_percentages = [label_counts[i] / total_pixels * 100 for i in range(num_colors)]

        return colors, color_percentages

    def get_contrast_color(self, hex_color):
        # Convert hex to RGB
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

        # Calculate luminance
        luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255

        # Choose black or white based on luminance
        if luminance > 0.5:
            return "#000000"  # Black for light backgrounds
        else:
            return "#FFFFFF"  # White for dark backgrounds

    def rgb_to_hsb(self, hex_color):
        # Convert hex to RGB
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

        # Convert RGB to HSB
        h, s, v = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)

        # Convert to degrees and percentages
        h = h * 360
        s = s * 100
        b = v * 100

        return h, s, b


class LS_GetMainColors:

    def __init__(self):
        self.NODE_NAME = 'Get Main Colors'

    @classmethod
    def INPUT_TYPES(self):
        size_list = ['custom']
        size_list.extend(load_custom_size())
        k_means_algorithm_list = ["lloyd", "elkan"]
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "k_means_algorithm": (k_means_algorithm_list,),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("preview_image", "color_1", "color_2", "color_3", "color_4", "color_5",)
    FUNCTION = 'get_main_colors'
    CATEGORY = '😺dzNodes/LayerUtility'

    def get_main_colors(self, image, k_means_algorithm):

        ret_images = []

        grid_width = 512
        grid_height = 128
        line_width = 5

        for i in range(len(image)):
            pil_img = tensor2pil(torch.unsqueeze(image[i], 0)).convert("RGB")
            blured_image = gaussian_blur(pil_img, (pil_img.width + pil_img.height) // 400)

            accuracy = 60 # Adjusts accuracy by changing number of iterations of the K-means algorithm
            num_colors = 5
            num_iterations = int(512 * (accuracy / 100))
            original_colors = self.interrogate_colors(
                pil2tensor(blured_image), num_colors=num_colors, algorithm=k_means_algorithm, mix_iter=num_iterations, random_state=0)

            main_colors = self.ndarrays_to_colorhex(original_colors)
            log(f"main_colors={main_colors}")
            # draw colors image
            ret_image = Image.new('RGB', size=(grid_width, grid_height * len(main_colors)), color="white")
            draw = ImageDraw.Draw(ret_image)

            for j in range(len(main_colors)):
                x1 = 0
                y1 = grid_height * j
                draw.rectangle((x1, y1, x1 + grid_width, y1 + grid_height), fill=main_colors[j], outline=main_colors[j])
            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), main_colors[0], main_colors[1], main_colors[2], main_colors[3], main_colors[4],)

    def ndarrays_to_colorhex(self, colors:list) -> list:
        return [RGB_to_Hex((int(color[0]), int(color[1]), int(color[2]))) for color in colors]

    def interrogate_colors(self, image:torch.Tensor, num_colors:int, algorithm:str, mix_iter:int, random_state:int) -> list:
        from sklearn.cluster import KMeans
        pixels = image.view(-1, image.shape[-1]).numpy()
        colors = (
                KMeans(
                    n_clusters=num_colors,
                    algorithm=algorithm,
                    max_iter=mix_iter,
                    random_state=random_state,
                )
                .fit(pixels)
                .cluster_centers_
                * 255
        )
        return colors


NODE_CLASS_MAPPINGS = {
    "LayerUtility: GetMainColors": LS_GetMainColors,
    "LayerUtility: GetMainColorsV2": LS_GetMainColorsV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: GetMainColors": "LayerUtility: Get Main Colors",
    "LayerUtility: GetMainColorsV2": "LayerUtility: Get Main Colors V2",

}