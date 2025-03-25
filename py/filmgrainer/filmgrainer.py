# Filmgrainer - by Lars Ole Pontoppidan - MIT License

from PIL import Image, ImageFilter
import os
import tempfile
import numpy as np

import filmgrainer.graingamma as graingamma
import filmgrainer.graingen as graingen


def _grainTypes(typ):
    # After rescaling to make different grain sizes, the standard deviation
    # of the pixel values change. The following values of grain size and power
    # have been imperically chosen to end up with approx the same standard 
    # deviation in the result:
    if typ == 1:
        return (0.8, 63) # more interesting fine grain
    elif typ == 2:
        return (1, 45) # basic fine grain
    elif typ == 3:
        return (1.5, 50) # coarse grain
    elif typ == 4:
        return (1.6666, 50) # coarser grain
    else:
        raise ValueError("Unknown grain type: " + str(typ))

# Grain mask cache
MASK_CACHE_PATH = os.path.join(tempfile.gettempdir(), "mask-cache")

def _getGrainMask(img_width:int, img_height:int, saturation:float, grayscale:bool, grain_size:float, grain_gauss:float, seed):
    if grayscale:
        str_sat = "BW"
        sat = -1.0 # Graingen makes a grayscale image if sat is negative
    else:
        str_sat = str(saturation)
        sat = saturation

    # filename = MASK_CACHE_PATH + "grain-%d-%d-%s-%s-%s-%d.png" % (
    #     img_width, img_height, str_sat, str(grain_size), str(grain_gauss), seed)
    # if os.path.isfile(filename):
    #     # print("Reusing: %s" % filename)
    #     mask = Image.open(filename)
    # else:
    #     mask = graingen.grainGen(img_width, img_height, grain_size, grain_gauss, sat, seed)
    #     # print("Saving: %s" % filename)
    #     if not os.path.isdir(MASK_CACHE_PATH):
    #         os.mkdir(MASK_CACHE_PATH)
    #     mask.save(filename, format="png", compress_level=1)
    mask = graingen.grainGen(img_width, img_height, grain_size, grain_gauss, sat, seed)
    return mask


def process(image:Image, scale:float, src_gamma:float, grain_power:float, shadows:float,
            highs:float, grain_type:int, grain_sat:float, gray_scale:bool, sharpen:int, seed:int):
            
    # image = np.clip(image, 0, 1)  # Ensure the values are within [0, 1]
    # image = (image * 255).astype(np.uint8)
    # img = Image.fromarray(image).convert("RGB")
    img = image
    org_width = img.size[0]
    org_height = img.size[1]
    
    if scale != 1.0:
        # print("Scaling source image ...")
        img = img.resize((int(org_width / scale), int(org_height / scale)),
                          resample = Image.LANCZOS)
    
    img_width = img.size[0]
    img_height = img.size[1]
    # print("Size: %d x %d" % (img_width, img_height))

    # print("Calculating map ...")
    map = graingamma.Map.calculate(src_gamma, grain_power, shadows, highs)
    # map.saveToFile("map.png")

    # print("Calculating grain stock ...")
    (grain_size, grain_gauss) = _grainTypes(grain_type)
    mask = _getGrainMask(img_width, img_height, grain_sat, gray_scale, grain_size, grain_gauss, seed)

    mask_pixels = mask.load()
    img_pixels = img.load()

    # Instead of calling map.lookup(a, b) for each pixel, use the map directly:
    lookup = map.map

    if gray_scale:
        # print("Film graining image ... (grayscale)")
        for y in range(0, img_height):
            for x in range(0, img_width):
                m = mask_pixels[x, y]
                (r, g, b) = img_pixels[x, y]
                gray = int(0.21*r + 0.72*g + 0.07*b)
                #gray_lookup = map.lookup(gray, m)
                gray_lookup = lookup[gray, m]
                img_pixels[x, y] = (gray_lookup, gray_lookup, gray_lookup)
    else:
        # print("Film graining image ...")
        for y in range(0, img_height):
            for x in range(0, img_width):
                (mr, mg, mb) = mask_pixels[x, y]
                (r, g, b) = img_pixels[x, y]
                r = lookup[r, mr]
                g = lookup[g, mg]
                b = lookup[b, mb]
                img_pixels[x, y] = (r, g, b)
    
    if scale != 1.0:
        # print("Scaling image back to original size ...")
        img = img.resize((org_width, org_height), resample = Image.LANCZOS)
    
    if sharpen > 0:
        # print("Sharpening image: %d pass ..." % sharpen)
        for x in range(sharpen):
            img = img.filter(ImageFilter.SHARPEN)

    return np.array(img).astype('float32') / 255.0