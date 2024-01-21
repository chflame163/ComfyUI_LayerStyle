# ComfyUI Layer Style
A set of nodes for ComfyUI that generate image like Adobe Photoshop's Layer Style.

The completed nodes are as follows, and further work is in progress.
![image](image/title.png)

[中文说明点这里](./README_CN.MD)

## Update
* Add GradientOverlay and ColorOverlay nodes. Add invalid mask input judgment and ignore it when invalid mask is input.
* Commit InnerGlow, InnerShadow and MotionBlur nodes.
* Renaming all completed nodes, the nodes are divided into 4 groups：LayerStyle, LayerMask, LayerUtility, LayerFilter. workflows containing old version nodes need to be manually replaced with new version nodes.
* OuterGlow node has undergone significant modifications by adding options for **_brightness_**, **_light_color_**, and **_glow_color_**.
* Commit MaskInvert node.
* Commit ColorPick node.
* Commit Stroke node.
* Commit MaskPreview node.
* Commit ImageOpacity node.
* The layer_mask is not a mandatory input now. it is allowed to use layers and masks with different shapes, but the size must be consistent.
* Commit ImageBlend node.
* Commit OuterGlow node.
* Commit DropShadow node.

# LayerStyle
![image](image/layerstyle_nodes.png)

### DropShadow
Generate shadow
![image](image/drop_shadow_example.png)


Node options:
![image](image/drop_shadow_node.png)
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1,2</sup>: Mask for layer_image, shadows are generated according to their shape.
* invert_mask: Whether to reverse the mask.
* blend_mode<sup>3</sup>: Blending mode of shadows.
* opacity: Opacity of shadow.
* distance_x: Horizontal offset of shadow.
* distance_y: Vertical offset of shadow.
* grow: Shadow expansion amplitude.
* blur: Shadow blur level.
* shadow_color<sup>4</sup>: Shadow color.

### OuterGlow
Generate outer glow
![image](image/outer_glow_example.png)

Node options:
![image](image/outer_glow_node.png)
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1,2</sup>: Mask for layer_image, grow are generated according to their shape.
* invert_mask: Whether to reverse the mask.
* blend_mode<sup>3</sup>: Blending mode of glow.
* opacity: Opacity of glow.
* brightness: Luminance of light.
* glow_range: range of glow.
* blur：blur of glow.
* light_color<sup>4</sup>: Center part color of glow.
* glow_color<sup>4</sup>: Edge part color of glow.

### InnerShadow
Generate inner shadow
![image](image/inner_shadow_example.png)

Node options:
![image](image/inner_shadow_node.png)
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1,2</sup>: Mask for layer_image, shadows are generated according to their shape.
* invert_mask: Whether to reverse the mask.
* blend_mode<sup>3</sup>: Blending mode of shadows.
* opacity: Opacity of shadow.
* distance_x: Horizontal offset of shadow.
* distance_y: Vertical offset of shadow.
* grow: Shadow expansion amplitude.
* blur: Shadow blur level.
* shadow_color<sup>4</sup>: Shadow color.

### InnerGlow
Generate inner glow
![image](image/inner_glow_example.png)

Node options:  
![image](image/inner_glow_node.png)
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1,2</sup>: Mask for layer_image, grow are generated according to their shape.
* invert_mask: Whether to reverse the mask.
* blend_mode<sup>3</sup>: Blending mode of glow.
* opacity: Opacity of glow.
* brightness: Luminance of light.
* glow_range: range of glow.
* blur：blur of glow.
* light_color<sup>4</sup>: Center part color of glow.
* glow_color<sup>4</sup>: Edge part color of glow.


### Stroke
Generate a stroke of layer。
![image](image/stroke_example.png)

Node options:   
![image](image/stroke_node.png)
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1,2</sup>: Mask for layer_image, stroke are generated according to their shape.
* invert_mask: Whether to reverse the mask.
* blend_mode<sup>3</sup>: Blending mode of stroke.
* opacity: Opacity of stroke.
* stroke_grow: Stroke expansion/contraction amplitude, positive values indicate expansion and negative values indicate contraction.
* stroke_width: Stroke width.
* stroke_color<sup>4</sup>: Stroke color, described in hexadecimal RGB format.

### GradientOverlay
Generate gradient overlay
![image](image/gradient_overlay_example.png)

Node options:   
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1,2</sup>: Mask for layer_image.
* invert_mask: Whether to reverse the mask.
* blend_mode<sup>3</sup>: Blending mode of gradient.
* opacity: Opacity of stroke.
* start_color: Color at the beginning of the gradient.
* start_alpha: Transparency at the beginning of the gradient.
* end_color: Color at the end of the gradient.
* end_alpha: Transparency at the end of the gradient.
* angle: Gradient rotation angle.

### ColorOverlay
Generate color overlay
![image](image/color_overlay_example.png)

Node options:
![image](image/color_overlay_node.png)
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1,2</sup>: Mask for layer_image.
* invert_mask: Whether to reverse the mask.
* blend_mode<sup>3</sup>: Blending mode of color.
* opacity: Opacity of stroke.
* color: Color of overlay.


#### Annotation for LayerStyle nodes:
<sup>1</sup>  The layer_image, layer_mask and the background_image(if have input), These three items must be of the same size.。

<sup>2</sup>  The mask not a mandatory input item. the alpha channel of the image is used by default. If the image input does not include an alpha channel, the entire image's alpha channel will be automatically created. if have masks input simultaneously, the alpha channel will be overwrite by the mask.

<sup>3</sup>  The blend mode include **_normal, multply, screen, add, subtract, difference, darker_** and **_lighter_**.

<sup>4</sup>  The RGB color described by hexadecimal RGB format, like '#FA3D86'.

# LayerMask
![image](image/layermask_nodes.png)

### MaskPreview:
Preview the input mask
![image](image/mask_invert.png)

### MaskInvert:
Invert the mask
![image](image/mask_invert_node.png)


# LayerUtility
![image](image/layerutility_nodes.png)

### ImageBlend:
Composit  layer image and background image, multiple blend modes are available for option, and transparency can be set.
![image](image/image_blend_example.png)

Node options:
![image](image/image_blend_node.png)
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1,2</sup>: Mask for layer_image.
* invert_mask: Whether to reverse the mask.
* blend_mode<sup>3</sup>: Blending mode.
* opacity: Opacity of blend.

### ImageOpacity:
Adjust image opacity
![image](image/image_opacity_example.png)

Node option:   
* image<sup>1</sup>: Image input, supporting RGB and RGBA. if is RGB, the alpha channel of the entire image will be automatically added.
* mask<sup>1,2</sup> : Mask input.
* invert_mask: Whether to reverse the mask.
* opacity: Opacity of image.

### ColorPicker:
Modify web extensions from [mtb nodes](https://github.com/melMass/comfy_mtb). Select colors on the color palette and output RGB values.
![image](image/color_picker.png)

Node options:
* mode： The output format is available in hexadecimal (HEX) and decimal (DEC).  

Output Type： 
* string

# LayerFilter
![image](image/layerfilter_nodes.png)

### MotionBlur:
Make the image motion blur
![image](image/motion_blur_example.png)

Node options:
* angle: The angle of blur.
* blur: The size of blur.

## Example workflow
image Some JSON workflow files in the workflow directory, that is example for ComfyUI.

## The following tasks are in the planned list

* LayerUtility:
imageblend advance
color image
gradient image

* LayerMask:
text mask
gradient mask
mask grow&blur
mask subtract
mask combine
mask diffrent
mask gradient
mask softedge

* LayerFilter:
bulr
emboss
contour
findedge


## How to install 

* Open the cmd window in the plugin directory of ComfyUI, like "ComfyUI\custom_nodes\"，type```git clone https://github.com/chflame163/ComfyUI_LayerStyle.git```
or download the zip file and extracted, copy the resulting folder to ComfyUI\custom_ Nodes\

* Install dependency packages, open the cmd window in the WordCloud plugin directory like "ComfyUI\custom_ Nodes\ComfyUI_WordCloud" and enter the following command:
```..\..\..\python_embeded\python.exe -m pip install -r requirements.txt```

* Restart ComfyUI