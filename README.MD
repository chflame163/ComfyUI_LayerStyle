# ComfyUI Layer Style
A set of nodes for ComfyUI that generate image like Adobe Photoshop's Layer Style.

![image](image/title.png)

Nodes are divided into four groups according to their functions: LayerStyle, LayerMask, LayerUtility and LayerFilter.
* [LayerStyle](#LayerStyle) nodes provides layer styles that mimic Adobe Photoshop.
* [LayerMask](#LayerMask) nodes provides mask assistance tools.
* [LayerUtility](#LayerUtility) nodes provides auxiliary nodes related to layer composit tools and workflows.
* [LayerFilter](#LayerFilter) nodes provides image effect filters.

[中文说明点这里](./README_CN.MD)

## Update
* Add [ImageBlendAdvance](#ImageBlendAdvance) node. This node allows for the synthesis of background images and layers of different sizes, providing a more free synthesis experience. 
Add [PrintInfo](#PrintInfo) node as a workflow debugging aid.
* Add [GradientOverlay](#GradientOverlay) and [ColorOverlay](#ColorOverlay) nodes. 
Add invalid mask input judgment and ignore it when invalid mask is input.
* Commit [InnerGlow](#InnerGlow), [InnerShadow](#InnerShadow) and [MotionBlur](#MotionBlur) nodes.
* Renaming all completed nodes, the nodes are divided into 4 groups：LayerStyle, LayerMask, LayerUtility, LayerFilter. workflows containing old version nodes need to be manually replaced with new version nodes.
* [OuterGlow](#OuterGlow) node has undergone significant modifications by adding options for **_brightness_**, **_light_color_**, and **_glow_color_**.
* Commit [MaskInvert](#MaskInvert) node.
* Commit [ColorPick](#ColorPick) node.
* Commit [Stroke](#Stroke) node.
* Commit [MaskPreview](#MaskPreview) node.
* Commit [ImageOpacity](#ImageOpacity) node.
* The layer_mask is not a mandatory input now. it is allowed to use layers and masks with different shapes, but the size must be consistent.
* Commit [ImageBlend](#ImageBlend) node.
* Commit [OuterGlow](#OuterGlow) node.
* Commit [DropShadow](#DropShadow) node.


# <a id="table1">LayerStyle</a>
![image](image/layerstyle_nodes.png)


### <a id="table1">DropShadow</a>
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
[note](#Annotation)


### <a id="table1">OuterGlow</a>
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
[note](#Annotation)


### <a id="table1">InnerShadow</a>
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
[note](#Annotation)


### <a id="table1">InnerGlow</a>
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
[note](#Annotation)


### <a id="table1">Stroke</a>
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
[note](#Annotation)


### <a id="table1">GradientOverlay</a>
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
[note](#Annotation)


### <a id="table1">ColorOverlay</a>
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
[note](#Annotation)




# <a id="table1">LayerMask</a>
![image](image/layermask_nodes.png)


### <a id="table1">MaskPreview</a>
Preview the input mask
![image](image/mask_invert.png)


### <a id="table1">MaskInvert</a>
Invert the mask
![image](image/mask_invert_node.png)


# <a id="table1">LayerUtility</a>
![image](image/layerutility_nodes.png)


### <a id="table1">ImageBlendAdvance</a>
Used for compositing layers, allowing for compositing layer images of different sizes on the background image, and setting positions and transformations. multiple mixing modes are available for selection, and transparency can be set.

The node provide layer transformation_methods and anti_aliasing options. helps improve the quality of synthesized images.

The node provides mask output that can be used for subsequent workflows.
![image](image/image_blend_advance_example.png)

Node options:   
![image](image/image_blend_advance_node.png)
* background_image: The background image.
* layer_image<sup>5</sup>: Layer image for composite.
* layer_mask<sup>2,5</sup>: Mask for layer_image.
* invert_mask: Whether to reverse the mask.
* blend_mode<sup>3</sup>: Blending mode.
* opacity: Opacity of blend.
* x_percent: Horizontal position of the layer on the background image, expressed as a percentage, with 0 on the far left and 100 on the far right. It can be less than 0 or more than 100, indicating that some of the layer's content is outside the screen.
* y_percent: Vertical position of the layer on the background image, expressed as a percentage, with 0 on the top and 100 on the bottom. For example, setting it to 50 indicates vertical center, 20 indicates upper center, and 80 indicates lower center.
* mirror: Mirror flipping. Provide two flipping modes, horizontal flipping and vertical flipping.
* scale: Layer magnification, 1.0 represents the original size.
* aspect_ratio: Layer aspect ratio. 1.0 is the original ratio, a value greater than this indicates elongation, and a value less than this indicates flattening.
* rotate: Layer rotation degree.
* Sampling methods for layer enlargement and rotation, including lanczos, bicubic, hamming, bilinear, box and nearest. Different sampling methods can affect the image quality and processing time of the synthesized image.
* anti_aliasing: Anti aliasing, ranging from 0 to 16, the larger the value, the less obvious the aliasing. An excessively high value will significantly reduce the processing speed of the node.
[note](#Annotation)

### <a id="table1">ImageBlend</a>
A simple node for composit layer image and background image, multiple blend modes are available for option, and transparency can be set.
![image](image/image_blend_example.png)

Node options:
![image](image/image_blend_node.png)
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1,2</sup>: Mask for layer_image.
* invert_mask: Whether to reverse the mask.
* blend_mode<sup>3</sup>: Blending mode.
* opacity: Opacity of blend.
[note](#Annotation)

### <a id="table1">ImageOpacity</a>
Adjust image opacity
![image](image/image_opacity_example.png)

Node option:   
* image<sup>5</sup>: Image input, supporting RGB and RGBA. if is RGB, the alpha channel of the entire image will be automatically added.
* mask<sup>2,5</sup> : Mask input.
* invert_mask: Whether to reverse the mask.
* opacity: Opacity of image.
[note](#Annotation)

### <a id="table1">ColorPicker</a>
Modify web extensions from [mtb nodes](https://github.com/melMass/comfy_mtb). Select colors on the color palette and output RGB values.
![image](image/color_picker.png)

Node options:
* mode： The output format is available in hexadecimal (HEX) and decimal (DEC).  

Output Type： 
* string

### <a id="table1">PrintInfo</a>
![image](image/print_info_node.png)
Used to provide assistance for workflow debugging. When running, the properties of any object connected to this node will be printed to the console.

This node allows any type of input.


# <a id="table1">LayerFilter</a>
![image](image/layerfilter_nodes.png)


### <a id="table1">MotionBlur</a>
Make the image motion blur
![image](image/motion_blur_example.png)

Node options:
* angle: The angle of blur.
* blur: The size of blur.


#### <a id="table1">Annotation</a> for nodes
<sup>1</sup>  The layer_image, layer_mask and the background_image(if have input), These three items must be of the same size.

<sup>2</sup>  The mask not a mandatory input item. the alpha channel of the image is used by default. If the image input does not include an alpha channel, the entire image's alpha channel will be automatically created. if have masks input simultaneously, the alpha channel will be overwrite by the mask.

<sup>3</sup>  The blend mode include **_normal, multply, screen, add, subtract, difference, darker_** and **_lighter_**.

<sup>4</sup>  The RGB color described by hexadecimal RGB format, like '#FA3D86'.

<sup>5</sup>  The layer_image and layer_mask must be of the same size.

## Example workflow
image Some JSON workflow files in the workflow directory, that is example for ComfyUI.

## The following tasks are in the planned list

 LayerUtility:
* color image
* gradient image
* image channel split
* image channel merge

LayerMask:
* text mask
* gradient mask
* mask grow&blur
* mask subtract
* mask combine
* mask diffrent
* mask gradient
* mask softedge

LayerFilter:
* bulr
* emboss
* contour
* findedge


## How to install 

* Open the cmd window in the plugin directory of ComfyUI, like "ComfyUI\custom_nodes\"，type```git clone https://github.com/chflame163/ComfyUI_LayerStyle.git```
or download the zip file and extracted, copy the resulting folder to ComfyUI\custom_ Nodes\

* Install dependency packages, open the cmd window in the WordCloud plugin directory like "ComfyUI\custom_ Nodes\ComfyUI_WordCloud" and enter the following command:
```..\..\..\python_embeded\python.exe -m pip install -r requirements.txt```

* Restart ComfyUI