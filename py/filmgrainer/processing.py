import cv2
import numpy as np

def generate_blurred_images(image, blur_strength, steps, focus_spread=1):
    blurred_images = []
    for step in range(1, steps + 1):
        # Adjust the curve based on the curve_weight
        blur_factor = (step / steps) ** focus_spread * blur_strength
        blur_size = max(1, int(blur_factor))
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1  # Ensure blur_size is odd
        
        # Apply Gaussian Blur
        blurred_image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
        blurred_images.append(blurred_image)
    return blurred_images

def apply_blurred_images(image, blurred_images, mask):
    steps = len(blurred_images)  # Calculate the number of steps based on the blurred images provided
    final_image = np.zeros_like(image)
    step_size = 1.0 / steps
    for i, blurred_image in enumerate(blurred_images):
        # Calculate the mask for the current step
        current_mask = np.clip((mask - i * step_size) * steps, 0, 1)
        next_mask = np.clip((mask - (i + 1) * step_size) * steps, 0, 1)
        blend_mask = current_mask - next_mask

        # Apply the blend mask
        final_image += blend_mask[:, :, np.newaxis] * blurred_image

    # Ensure no division by zero; add the original image for areas without blurring
    final_image += (1 - np.clip(mask * steps, 0, 1))[:, :, np.newaxis] * image
    return final_image