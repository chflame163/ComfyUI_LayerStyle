from .imagefunc import *

any = AnyType("*")

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
    "LayerUtility: GetMainColors": LS_GetMainColors
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: GetMainColors": "LayerUtility: Get Main Colors"
}