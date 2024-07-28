# ComfyUI Layer Style

[中文说明点这里](./README_CN.MD)    

商务合作请联系email [chflame@163.com](mailto:chflame@163.com).

For business cooperation, please contact email [chflame@163.com](mailto:chflame@163.com).

A set of nodes for ComfyUI that can composite layer and mask to achieve Photoshop like functionality.  

It migrate some basic functions of PhotoShop to ComfyUI, aiming to centralize the workflow and reduce the frequency of software switching.  

![image](image/title.jpg)    
<font size="1">*this workflow (title_example_workflow.json) is in the workflow directory.  </font><br /> 



## Example workflow
Some JSON workflow files in the	```workflow``` directory, That's examples of how these nodes can be used in ComfyUI.


## How to install 
(Taking ComfyUI official portable package and Aki ComfyUI package as examples, please modify the dependency environment directory for other ComfyUI environments)

### Install plugin
* Recommended use ComfyUI Manager for installation.

* Or open the cmd window in the plugin directory of ComfyUI, like ```ComfyUI\custom_nodes```，type    
```
git clone https://github.com/chflame163/ComfyUI_LayerStyle.git
```
* Or download the zip file and extracted, copy the resulting folder to ```ComfyUI\custom_ Nodes```    

### Install dependency packages 
* for ComfyUI official portable package, double-click the ```install_requirements.bat``` in the plugin directory, for Aki ComfyUI package double-click on the ```install_requirements_aki.bat``` in the plugin directory, and wait for the installation to complete.
  
* Or install dependency packages, open the cmd window in the ComfyUI_LayerStyle plugin directory like 
```ComfyUI\custom_ Nodes\ComfyUI_LayerStyle``` and enter the following command,

&emsp;&emsp;for ComfyUI official portable package, type:

```
..\..\..\python_embeded\python.exe -s -m pip install .\whl\docopt-0.6.2-py2.py3-none-any.whl
..\..\..\python_embeded\python.exe -s -m pip install -r requirements.txt
.\repair_dependency.bat
```
&emsp;&emsp;for Aki ComfyUI package, type:

```
..\..\python\python.exe -s -m pip install .\whl\docopt-0.6.2-py2.py3-none-any.whl
..\..\python\python.exe -s -m pip install -r requirements.txt
.\repair_dependency.bat
```
* Restart ComfyUI.

## Common Issues
If the node cannot load properly or there are errors during use, please check the error message in the ComfyUI terminal window. The following are common errors and their solutions.

### Warning: xxxx.ini not found, use default xxxx..
This warning message indicates that the ini file cannot be found and does not affect usage. If you do not want to see these warnings, please modify all ```*.ini.example``` files in the plugin directory to ```*.ini```.

### ModuleNotFoundError: No module named 'psd_tools'
This error is that the ```psd_tools``` were not installed correctly.   

Solution:
* Close ComfyUI and open the terminal window in the plugin directory and execute the following command:
```../../../python_embeded/python.exe -s -m pip install psd_tools```
If error occurs during the installation of psd_tool, such as ```ModuleNotFoundError: No module named 'docopt'``` , please download [docopt's whl](https://www.piwheels.org/project/docopt/) and manual install it. 
execute the following command in terminal window:
```../../../python_embeded/python.exe -s -m pip install path/docopt-0.6.2-py2.py3-none-any.whl``` the ```path``` is path name of whl file.

### Cannot import name 'guidedFilter' from 'cv2.ximgproc'
This error is caused by incorrect version of the ```opencv-contrib-python``` package，or this package is overwriteen by other opencv packages. 


### NameError: name 'guidedFilter' is not defined
The reason for the problem is the same as above.

### Cannot import name 'VitMatteImageProcessor' from 'transformers' 
This error is caused by the low version of ```transformers``` package. 


### insightface Loading very slow
This error is caused by the low version of ```protobuf``` package. 

#### For the issues with the above three dependency packages, please double click ```repair_dependency.bat``` (for Official ComfyUI Protable) or  ```repair_dependency_aki.bat``` (for ComfyUI-aki-v1.x) in the plugin folder to automatically fix them.

### ValueError: Trimap did not contain foreground values (xxxx...)
This error is caused by the mask area being too large or too small when using the ```PyMatting``` method to handle the mask edges.    

Solution:
* Please adjust the parameters to change the effective area of the mask. Or use other methods to handle the edges.

### Requests.exceptions.ProxyError: HTTPSConnectionPool(xxxx...)
When this error has occurred, please check the network environment.


## Update
<font size="4">**If the dependency package error after updating, please reinstall the relevant dependency packages. </font><br />    

* Commit [Florence2Ultra](#Florence2Ultra),  [Florence2Image2Prompt](#Florence2Image2Prompt) and [LoadFlorence2Model](#LoadFlorence2Model) nodes.
* [TransparentBackgroundUltra](#TransparentBackgroundUltra) node add new model support. Please download the model file according to the instructions.
* Commit [SegformerUltraV2](#SegformerUltraV2), [SegfromerFashionPipeline](#SegfromerFashionPipeline) and [SegformerClothesPipeline](#SegformerClothesPipeline) nodes, used for segmentation of clothing. please download the model file according to the instructions.
* Commit ```install_requirements.bat``` and ```install_requirements_aki.bat```, One click solution to install dependency packages.
* Commit [TransparentBackgroundUltra](#TransparentBackgroundUltra) node, it remove background based on transparent-background model.
* Change the VitMatte model of the [Ultra](#Ultra) node to a local call. Please download [all files of vitmatte model](https://huggingface.co/hustvl/vitmatte-small-composition-1k/tree/main) to the ```ComfyUI/models/vitmatte``` folder.
* [GetColorToneV2](#GetColorToneV2) node add the ```mask``` method to the color selection option, which can accurately obtain the main color and average color within the mask.
* [ImageScaleByAspectRatioV2](#ImageScaleByAspectRatioV2) node add the "background_color" option.
* [LUT Apply](#LUT) Add the "strength" option.
* Commit [AutoAdjustV2](#AutoAdjustV2) node, add optional mask input and support for multiple automatic color adjustment modes.
* Due to the upcoming discontinuation of gemini-pro vision services, [PromptTagger](#PromptTagger) and [PromptEmbellish](#PromptEmbellish) have added the "gemini-1.5-flash" API to continue using it.
* [Ultra](#Ultra) nodes added the option to run ```VitMatte``` on the CUDA device, resulting in a 5-fold increase in running speed.
* Commit [QueueStop](#QueueStop) node, used to terminate the queue operation.
* Optimize performance of the ```VitMate``` method for [Ultra](#Ultra) nodes when processing large-size image.
* [CropByMaskV2](#CropByMaskV2) add option to round the cutting size by multiples.
* Commit [CheckMask](#CheckMask) node, it detect whether the mask contains sufficient effective areas. Commit [HSVValue](#HSVValue) node, it convert color values to HSV values.
* [BooleanOperatorV2](#BooleanOperatorV2), [NumberCalculatorV2](#NumberCalculatorV2), [Integer](#Integer), [Float](#Float), [Boolean](#Boolean) nodes add string output to output the value as a string for use with [SwitchCase](#SwitchCase).
* Commit [SwitchCase](#SwitchCase) node, Switches the output based on the matching string. Can be used for any type of data switching.
* Commit [String](#String) node, Used to output a string. It is the TextBox simplified node.
* Commit [If](#If) node，Switches output based on Boolean conditional input. Can be used for any type of data switching.
* Commit [StringCondition](#StringCondition) node, Determines whether the text contains or does not contain a substring.
* Commit [NumberCalculatorV2](#NumberCalculatorV2) node，Add the nth root operation. Commit [BooleanOperatorV2](#BooleanOperatorV2) node, Increasing greater/less than, greater/less then or equal logical judgment. The two nodes can access numeric inputs and can input numeric values within the node. Note: Numeric input takes precedence. Values in nodes will not be valid when there is input.
* Commit [SD3NegativeConditioning](#SD3NegativeConditioning) node, Encapsulate the four nodes of Negative Condition in SD3 into a separate node.
* [ImageRemoveAlpha](#ImageRemoveAlpha) node add optional mask input.
* Commit [HLFrequencyDetailRestore](#HLFrequencyDetailRestore) node, Using low-frequency filtering and high-frequency preserving to restore image details, the fusion is better.
* Commit [AddGrain](#AddGrain) and [MaskGrain](#MaskGrain) nodes, Add noise to a picture or mask.
* Commit [FilmV2](#FilmV2) node, The fastgrain method is added on the basis of the previous one, and the noise generation speed is 10 times faster.
* Commit [ImageToMask](#ImageToMask) node, it can be converted image into mask. Supports converting any channel in LAB, RGBA, YUV, and HSV modes into masks, while providing color scale adjustment. Support mask optional input to obtain masks that only include valid parts.
* The blackpoint and whitepoint options in some nodes have been changed to slider adjustment for a more intuitive display. Include [MaskEdgeUltraDetailV2](#MaskEdgeUltraDetailV2), [SegmentAnythingUltraV2](#SegmentAnythingUltraV2), [RmBgUltraV2](#RmBgUltraV2)，[PersonMaskUltraV2](#PersonMaskUltraV2)，[BiRefNetUltra](#BiRefNetUltra), [SegformerB2ClothesUltra](#SegformerB2ClothesUltra), [BlendIfMask](#BlendIfMask) and [Levels](#Levels).
* [ImageScaleRestoreV2](#ImageScaleRestoreV2) and  [ImageScaleByAspectRatioV2](#ImageScaleByAspectRatioV2) nodes add the ```total_pixel``` method to scale images.
* Commit [MediapipeFacialSegment](#MediapipeFacialSegment) node，Used to segment facial features, including left and right eyebrows, eyes, lips, and teeth.
* Commit [BatchSelector](#BatchSelector) node，Used to retrieve specified images or masks from batch images or masks.
* LayerUtility creates new subdirectories such as SystemIO, Data, and Prompt. Some nodes are classified into subdirectories.
* Commit [MaskByColor](#MaskByColor) node, Generate a mask based on the selected color.
* Commit [LoadPSD](#LoadPSD) node, It read the psd format, and output layer images. Note that this node requires the installation of the ```psd_tools``` dependency package, If error occurs during the installation of psd_tool, such as ```ModuleNotFoundError: No module named 'docopt'``` , please download [docopt's whl](https://www.piwheels.org/project/docopt/) and manual install it. 
* Commit [SegformerB2ClothesUltra](#SegformerB2ClothesUltra) node, it used to segment character clothing. The model segmentation code is from[StartHua](https://github.com/StartHua/Comfyui_segformer_b2_clothes),  thanks to the original author.
* [SaveImagePlus](#SaveImagePlus) node adds the output workflow to the json function, supports ```%date``` and ```%time``` to embeddint date or time to path and filename, and adds the preview switch.
* Commit [SaveImagePlus](#SaveImagePlus) node，It can customize the directory where the picture is saved, add a timestamp to the file name, select the save format, set the image compression rate, set whether to save the workflow, and optionally add invisible watermarks to the picture.
* Commit [AddBlindWaterMark](#AddBlindWaterMark), [ShowBlindWaterMark](#ShowBlindWaterMark) nodes, Add invisible watermark and decoded watermark to the picture. Commit [CreateQRCode](#CreateQRCode), [DecodeQRCode](#DecodeQRCode) nodes, It can generate two-dimensional code pictures and decode two-dimensional codes.
* [ImageScaleRestoreV2](#ImageScaleRestoreV2), [ImageScaleByAspectRatioV2](#ImageScaleByAspectRatioV2), [ImageAutoCropV2](#ImageAutoCropV2) nodes add options for ```width``` and ```height```, which can specify width or height as fixed values.
* Commit [PurgeVRAM](#PurgeVRAM) node, Clean up VRAM an RAM.
* Commit [AutoAdjust](#AutoAdjust) node, it can automatically adjust image contrast and white balance.
* Commit [RGBValue](#RGBValue) node to output the color value as a single decimal value of R, G, B. This idea is from [vxinhao](https://github.com/vxinhao/color2rgb), Thanks.
* Commit [seed](#seed) node to output the seed value. The [ImageMaskScaleAs](#ImageMaskScaleAs), [ImageScaleBySpectRatio](#ImageScaleBySpectRatio), [ImageScaleBySpectRatioV2](#ImageScaleBySpectRatioV2), [ImageScaleRestore](#ImageScaleRestore), [ImageScaleRestoreV2](#ImageScaleRestoreV2) nodes increase ```width```, ```height``` output.
* Commit [Levels](#Levels) node, it can achieve the same color levels adjustment function as Photoshop.[Sharp&Soft](#Sharp) add the "None" option.
* Commit [BlendIfMask](#BlendIfMask) node, This node cooperates with ImgaeBlendV2 or ImageBlendAdvanceV2 to achieve the same Blend If function as Photoshop.
* Commit [ColorTemperature](#ColorTemperature) and [ColorBalance](#ColorBalance) nodes, used to adjust the color temperature and color balance of the picture.
* Add new types of [Blend Mode V2](#BlendModeV2) between images. now supports up to 30 blend modes. The new blend mode is available for all V2 versions that support mixed mode nodes, including ImageBlend V2, ImageBlendAdvance V2, DropShadow V2, InnerShadow V2, OuterGlow V2, InnerGlow V2, Stroke V2, ColorOverlay V2, GradientOverlay V2.    
Part of the code for BlendMode V2 is from [Virtuoso Nodes for ComfyUI](https://github.com/chrisfreilich/virtuoso-nodes). Thanks to the original authors.
* Commit [YoloV8Detect](#YoloV8Detect) node.
* Commit [QWenImage2Prompt](#QWenImage2Prompt) node, this node is repackage of the [ComfyUI_VLM_nodes](https://github.com/gokayfem/ComfyUI_VLM_nodes)'s ```UForm-Gen2 Qwen Node```,  thanks to the original author.
* Commit [BooleanOperator](#BooleanOperator), [NumberCalculator](#NumberCalculator), [TextBox](#TextBox), [Integer](#Integer), [Float](#Float), [Boolean](#Boolean)nodes. These nodes can perform mathematical and logical operations.
* Commit [ExtendCanvasV2](#ExtendCanvasV2) node，support color value input.
* Commit [AutoBrightness](#AutoBrightness) node，it can automatically adjust the brightness of image.
* [CreateGradientMask](#CreateGradientMask) node add ```center``` option.
* Commit [GetColorToneV2](#GetColorToneV2) node, can select the main and average colors for the background or body. 
* Commit [ImageRewardFilter](#ImageRewardFilter) node, can filter out poor quality pictures.
* [Ultra](#Ultra) nodes add ```VITMatte(local)``` method, You can choose this method to avoid accessing huggingface.co if you have already downloaded the model before.
* Commit [HDR Effect](#HDR) node，it enhances the dynamic range and visual appeal of input images.  this node is repackage of [HDR Effects (SuperBeasts.AI)](https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts).
* Commit [CropBoxResolve](#CropBoxResolve) node.
* Commit [BiRefNetUltra](#BiRefNetUltra) node, it using the BiRefNet model to remove background has better recognition ability and ultra-high edge details.
* Commit [ImageAutoCropV2](#ImageAutoCropV2) node, it can choose not to remove the background, support mask input, and scale by long or short side size.
* Commit [ImageHub](#ImageHub) node, supports up to 9 sets of Image and Mask switching output, and supports random output.
* Commit [TextJoin](#TextJoin) node.
* Commit [PromptEmbellish](#PromptEmbellish) node. it output polished prompt words, and support inputting images as references.
* [Ultra](#Ultra) nodes have been fully upgraded to V2 version, with the addition of VITMatte edge processing method, which is suitable for handling semi transparent areas. Include [MaskEdgeUltraDetailV2](#MaskEdgeUltraDetailV2), [SegmentAnythingUltraV2](#SegmentAnythingUltraV2), [RmBgUltraV2](#RmBgUltraV2) and [PersonMaskUltraV2](#PersonMaskUltraV2) nodes.
* Commit [Color of Shadow & Highlight](#Highlight) node, it can adjust the color of the dark and bright parts separately. Commit [Shadow & Highlight Mask](#Shadow) node, it can output mask for dark and bright areas.
* Commit [CropByMaskV2](#CropByMaskV2) node, On the basis of the original node, it supports ```crop_box``` input, making it convenient to cut layers of the same size.
* Commit [SimpleTextImage](#SimpleTextImage) node, it generate simple typesetting images and masks from text. This node references some of the functionalities and code of [ZHO-ZHO-ZHO/ComfyUI-Text_Image-Composite](https://github.com/ZHO-ZHO-ZHO/ComfyUI-Text_Image-Composite).
* Commit [PromptTagger](#PromptTagger) node，Inference the prompts based on the image. and it can replace key word for the prompt(need apply for Google Studio API key). Upgrade [ColorImageV2](#ColorImageV2) and [GradientImageV2](#GradientImageV2)，support user customize preset sizes and size_as input.
* Commit [LaMa](#LaMa) node, it can erase objects from the image based on the mask. this node is repackage of [IOPaint](https://www.iopaint.com).
* Commit [ImageRemoveAlpha](#ImageRemoveAlpha) and [ImageCombineAlpha](#ImageCombineAlpha) nodes, alpha channel of the image can be removed or merged.
* Commit [ImageScaleRestoreV2](#ImageScaleRestoreV2) and [ImageScaleByAspectRatioV2](#ImageScaleByAspectRatioV2) nodes, supports scaling images to specified long or short edge sizes.
* Commit [PersonMaskUltra](#PersonMaskUltra) node, Generate masks for portrait's face, hair, body skin, clothing, or accessories. the model code for this node comes from [a-person-mask-generator](https://github.com/djbielejeski/a-person-mask-generator).
* Commit [LightLeak](#LightLeak) node, this filter simulate the light leakage effect of the film.
* Commit [Film](#Film) node, this filter simulate the grain, dark edge, and blurred edge of the film, support input depth map to simulate defocus. it is reorganize and encapsulate of [digitaljohn/comfyui-propost](https://github.com/digitaljohn/comfyui-propost).
* Commit [ImageAutoCrop](#ImageAutoCrop) node, which is designed to generate image materials for training models.
* Commit [ImageScaleByAspectRatio](#ImageScaleByAspectRatio) node, it can be scaled image or mask according to frame ratio.
* Fix the bug of color gradation in [LUT Apply](#LUT) node rendering, and this node now support for log color space. *Please load the dedicated log lut file for the log color space image.
* Commit [CreateGradientMask](#CreateGradientMask) node. Commit [LayerImageTransform](#LayerImageTransform) and [LayerMaskTransform](#LayerMaskTransform) nodes.
* Commit [MaskEdgeUltraDetail](#MaskEdgeUltraDetail) node, it process rough masks to ultra fine edges.Commit [Exposure](#Exposure) node.
* Commit [Sharp & Soft](#Sharp) node, it can enhance or smooth out image details. Commit [MaskByDifferent](#MaskByDifferent) node, it compare two images and output a Mask. Commit [SegmentAnythingUltra](#SegmentAnythingUltra) node, Improve the quality of mask edges. *If SegmentAnything is not installed, you will need to manually download the model.
* All nodes have fully supported batch images, providing convenience for video creation.
(The CropByMask node only supports cuts of the same size. if a batch mask_for_crop inputted, the data from the first sheet will be used.)
* Commit [RemBgUltra](#RemBgUltra) and [PixelSpread](#PixelSpread) nodes significantly improved mask quality. *RemBgUltra requires manual model download.
* Commit [TextImage](#TextImage) node, it generate text images and masks.
* Add new types of [blend mode](#Blend) between images. now supports up to 19 blend modes. add **color_burn, color_dodge, linear_burn, linear_dodge, overlay, soft_light, hard_light, vivid_light, pin_light, linear_light** and **hard_mix**. 
The newly added blend mode is applicable to all nodes that support blend mode.
* Commit [ColorMap](#ColorMap) filter node to create a pseudo color heatmap effect.
* Commit [WaterColor](#WaterColor) and [SkinBeauty](#SkinBeauty) nodes。These are image filters that generate watercolor and skin smoothness effects.
* Commit [ImageShift](#ImageShift)  node to shift the image and output a displacement seam mask, making it convenient to create continuous textures.
* Commit [ImageMaskScaleAs](#ImageMaskScaleAs) node to adjust the image or mask size based on the reference image.
* Commit [ImageScaleRestore](#ImageScaleRestore) node to work with CropByMask for local upscale and repair works.
* Commit [CropByMask](#CropByMask) and [RestoreCropBox](#RestoreCropBox) nodes. The combination of these two can partially crop and redraw the image before restoring it.
* Commit [ColorAdapter](#ColorAdapter) node, that can automatically adjust the color tone of the image.
* Commit [MaskStroke](#MaskStroke) node, it can generate mask contour strokes.
* Add [LayerColor](#LayerColor) node group, used to adjust image color. it include [LUT Apply](#LUT), [Gamma](#Gamma), [Brightness & Contrast](#Brightness), [RGB](#RGB), [YUV](#YUV), [LAB](#LAB) adn [HSV](#HSV).
* Commit [ImageChannelSplit](#ImageChannelSplit) and [ImageChannelMerge](#ImageChannelMerge) nodes.
* Commit [MaskMotionBlur](#MaskMotionBlur) node.
* Commit [SoftLight](#SoftLight) node.
* Commit [ChannelShake](#ChannelShake) node, that is filter, can produce channel dislocation effect similar like Tiktok logo.
* Commit [MaskGradient](#MaskGradient) node, can create a gradient in the mask.
* Commit [GetColorTone](#GetColorTone) node, can obtain the main color or average color of the image. 
Commit [MaskGrow](#MaskGrow) and [MaskEdgeShrink](#MaskEdgeShrink) nodes.
* Commit [MaskBoxDetect](#MaskBoxDetect) node, which can automatically detect the position through the mask and output it to the composite node.
Commit [XY to Percent](#Percent) node to convert absolute coordinates to percent coordinates.
Commit [GaussianBlur](#GaussianBlur) node.
Commit [GetImageSize](#GetImageSize) node.
* Commit [ExtendCanvas](#ExtendCanvas) node.
* Commit [ImageBlendAdvance](#ImageBlendAdvance) node. This node allows for the synthesis of background images and layers of different sizes, providing a more free synthesis experience. 
Commit [PrintInfo](#PrintInfo) node as a workflow debugging aid.
* Commit [ColorImage](#ColorImage) and [GradientImage](#GradientImage) nodes, Used to generate solid and gradient color images.
* Commit [GradientOverlay](#GradientOverlay) and [ColorOverlay](#ColorOverlay) nodes. 
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


## Description
Nodes are divided into 5 groups according to their functions: LayerStyle, LayerColor, LayerMask, LayerUtility and LayerFilter.
* [LayerStyle](#LayerStyle) nodes provides layer styles that mimic Adobe Photoshop.
![image](image/menu_layer_style.jpg)    
* [LayerColor](#LayerColor) node group provides color adjustment functionality.
![image](image/menu_layer_color.jpg)    
* [LayerMask](#LayerMask) nodes provides mask assistance tools.
![image](image/menu_layer_mask.jpg)    
* [LayerUtility](#LayerUtility) nodes provides auxiliary nodes related to layer composit tools and workflows.
![image](image/menu_layer_utility.jpg)    
* [LayerFilter](#LayerFilter) nodes provides image effect filters.
![image](image/menu_layer_filter.jpg)    

# <a id="table1">LayerStyle</a>
![image](image/layerstyle_title.jpg)    
![image](image/layerstyle_nodes.jpg)    


### <a id="table1">DropShadow</a>
Generate shadow
![image](image/drop_shadow_example.jpg)    

Node options:
![image](image/drop_shadow_node.jpg)    
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
* [note](#notes)


### <a id="table1">OuterGlow</a>
Generate outer glow
![image](image/outer_glow_example.jpg)    

Node options:
![image](image/outer_glow_node.jpg)    
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
* [note](#notes)


### <a id="table1">InnerShadow</a>
Generate inner shadow
![image](image/inner_shadow_example.jpg)    

Node options:
![image](image/inner_shadow_node.jpg)    
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
* [note](#notes)


### <a id="table1">InnerGlow</a>
Generate inner glow
![image](image/inner_glow_example.jpg)    

Node options:  
![image](image/inner_glow_node.jpg)    
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
* [note](#notes)


### <a id="table1">Stroke</a>
Generate a stroke of layer。
![image](image/stroke_example.jpg)    

Node options:   
![image](image/stroke_node.jpg)    
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1,2</sup>: Mask for layer_image, stroke are generated according to their shape.
* invert_mask: Whether to reverse the mask.
* blend_mode<sup>3</sup>: Blending mode of stroke.
* opacity: Opacity of stroke.
* stroke_grow: Stroke expansion/contraction amplitude, positive values indicate expansion and negative values indicate contraction.
* stroke_width: Stroke width.
* blur: Blur of stroke.
* stroke_color<sup>4</sup>: Stroke color, described in hexadecimal RGB format.
* [note](#notes)


### <a id="table1">GradientOverlay</a>
Generate gradient overlay
![image](image/gradient_overlay_example.jpg)    

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
* [note](#notes)


### <a id="table1">ColorOverlay</a>
Generate color overlay
![image](image/color_overlay_example.jpg)    

Node options:
![image](image/color_overlay_node.jpg)    
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1,2</sup>: Mask for layer_image.
* invert_mask: Whether to reverse the mask.
* blend_mode<sup>3</sup>: Blending mode of color.
* opacity: Opacity of stroke.
* color: Color of overlay.
* [note](#notes)

# <a id="table1">LayerColor</a>
![image](image/layercolor_title.jpg)    
![image](image/layercolor_nodes.jpg)    

### <a id="table1">LUT</a> Apply
Apply LUT to the image. only supports .cube format.
![image](image/lut_apply_example.jpg)    

Node options:  
![image](image/lut_apply_node.jpg)    
* LUT<sup>*</sup>: Here is a list of available. cube files in the LUT folder, and the selected LUT files will be applied to the image.
* color_space: For regular image, please select linear, for image in the log color space, please select log.
* strength: Range 0~100, LUT application strength. The larger the value, the greater the difference from the original image, and the smaller the value, the closer it is to the original image.

<sup>*</sup><font size="3">LUT folder is defined in ```resource_dir.ini```, this file is located in the root directory of the plug-in, and the default name is ```resource_dir.ini.example```. to use this file for the first time, you need to change the file suffix to ```.ini```.
Open the text editing software and find the line starting with "LUT_dir=", after "=", enter the custom folder path name. all .cube files in this folder will be collected and displayed in the node list during ComfyUI initialization.
If the folder set in ini is invalid, the LUT folder that comes with the plugin will be enabled.</font>

### <a id="table1">AutoAdjust</a>
Automatically adjust the brightness, contrast, and white balance of the image. Provide some manual adjustment options to compensate for the shortcomings of automatic adjustment.
![image](image/auto_adjust_example.jpg)    

Node Options:  
![image](image/auto_adjust_node.jpg)    
* strength: Strength of adjust. The larger the value, the greater the difference from the original image.
* brightness: Manual adjustment of brightness.
* contrast: Manual adjustment of contrast.
* saturation: Manual adjustment of saturation.
* red: Manual adjustment of the red channel.
* green: Manual adjustment of the green channel.
* blue: Manual adjustment of the blue channel.

### <a id="table1">AutoAdjustV2</a>
On the basis of AutoAdjust, add mask input and only calculate the content inside the mask for automatic color adjustment. Add multiple automatic adjustment modes.
![image](image/auto_adjust_v2_example.jpg)    

The following changes have been made based on AutoAdjust: 
![image](image/auto_adjust_v2_node.jpg)    
* mask: Optional mask input.
* mode: Automatic adjustment mode. "RGB" automatically adjusts according to the three channels of RGB, "lum + sat"automatically adjusts according to luminance and saturation, "luminance" automatically adjusts according to luminance, "saturation" automatically adjusts according to saturation, and "mono" automatically adjusts according to grayscale and outputs monochrome.

### <a id="table1">AutoBrightness</a>
Automatically adjust too dark or too bright image to moderate brightness, and support mask input. When  mask input, only the content of the mask part is used as the data source of the automatic brightness. The output is still the whole adjusted image.
![image](image/auto_brightness_example.jpg)    

Node options:  
![image](image/auto_brightness_node.jpg)    
* strength: Automatically adjust the intensity of the brightness. The larger the value, the more biased towards the middle value, the greater the difference from the original picture.
* saturation: Color saturation. Changes in brightness usually result in changes in color saturation, where appropriate compensation can be adjusted.

### <a id="table1">ColorAdapter</a>
Auto adjust the color tone of the image to resemble the reference image.
![image](image/color_adapter_example.jpg)    

Node options:  
![image](image/color_adapter_node.jpg)    
* opacity: The opacity of an image after adjusting its color tone.

### <a id="table1">Exposure</a>
Change the exposure of the image.
![image](image/exposure_example.jpg)    

### Color of Shadow & <a id="table1">Highlight</a>
Adjust the color of the dark and bright parts of the image.
![image](image/color_of_shadow_and_highlight_example.jpg)    

Node options:  
![image](image/color_of_shadow_and_highlight_node.jpg)    
* image: The input image.
* mask: Optional input. if there is input, only the colors within the mask range will be adjusted.
* shadow_brightness: The brightness of the dark area.
* shadow_saturation: The color saturation in the dark area.
* shadow_hue: The color hue in the dark area.
* shadow_level_offset: The offset of values in the dark area, where larger values bring more areas closer to the bright into the dark area.
* shadow_range: The transitional range of the dark area.
* highlight_brightness:  The brightness of the highlight area.
* highlight_saturation: The color saturation in the highlight area.
* highlight_hue: The color hue in the highlight area.
* highlight_level_offset: The offset of values in the highlight area, where larger values bring more areas closer to the dark into the highlight area.
* highlight_range: The transitional range of the highlight area.

Node option:  
* exposure: Exposure value. Higher values indicate brighter image.


### <a id="table1">ColorTemperature</a>
![image](image/color_temperature_example.jpg)    
Change the color temperature of the image.

Node Options:  
![image](image/color_temperature_node.jpg)    
* temperature: Color temperature value. Range between-100 and 100. The higher the value, the higher the color temperature (bluer); The lower the color temperature, the lower the color temperature (yellowish).

### <a id="table1">Levels</a>
![image](image/levels_example.jpg)    
Change the levels of image.

Node Options:  
![image](image/levels_node.jpg)    
* channel: Select the channel you want to adjust. Available in RGB, red, green, blue.
* black_point<sup>*</sup>: Input black point value. Value range 0-255, default 0.
* white_point<sup>*</sup>: Input white point value. Value range 0-255, default 255.
* gray_point: Input grey point values. Value range 0.01-9.99, default 1.
* output_black_point<sup>*</sup>: Output black point value. Value range 0-255, default 0.
* output_white_point<sup>*</sup>: Output white point value. Value range 0-255, default 255.

<sup>*</sup><font size="3">If the black_point or output_black_point value is greater than white_point or output_white_point, the two values are swapped, with the larger value used as white_point and the smaller value used as black_point.</font>

### <a id="table1">ColorBalance</a>
![image](image/color_balance_example.jpg)    
Change the color balance of an image.

Node Options:  
![image](image/color_balance_node.jpg)    
* cyan_red: Cyan-Red balance. negative values are leaning cyan, positive values are leaning red.
* magenta_green: Megenta-Green balance. negative values are leaning megenta, positive values are leaning green.
* yellow_blue: Yellow-Blue balance. negative values are leaning yellow, positive values are leaning blue.


### <a id="table1">Gamma</a>
Change the gamma value of the image.

Node options:  
![image](image/gamma_node.jpg)    
* gamma: Value of the Gamma.

### <a id="table1">Brightness</a> & Contrast
Change the brightness, contrast, and saturation of the image.

Node options:  
![image](image/brightness_&_contrast_node.jpg)    
* brightness: Value of brightness.
* contrast: Value of contrast.
* saturation: Value of saturation.

### <a id="table1">RGB</a>
Adjust the RGB channels of the image.

Node options:  
![image](image/RGB_node.jpg)    
* R: R channel.
* G: G channel.
* B: B channel.

### <a id="table1">YUV</a>
Adjust the YUV channels of the image.

Node options:  
![image](image/YUV_node.jpg)    
* Y: Y channel.
* U: U channel.
* V: V channel.

### <a id="table1">LAB</a>
Adjust the LAB channels of the image.

Node options:  
![image](image/LAB_node.jpg)    
* L: L channel.
* A: A channel.
* B: B channel.

### <a id="table1">HSV</a>
Adjust the HSV channels of the image.

Node options:  
![image](image/HSV_node.jpg)    
* H: H channel.
* S: S channel.
* V: V channel.


# <a id="table1">LayerUtility</a>
![image](image/layerutility_nodes.jpg)    


### <a id="table1">ImageBlendAdvance</a>
Used for compositing layers, allowing for compositing layer images of different sizes on the background image, and setting positions and transformations. multiple mixing modes are available for selection, and transparency can be set.

The node provide layer transformation_methods and anti_aliasing options. helps improve the quality of synthesized images.

The node provides mask output that can be used for subsequent workflows.
![image](image/image_blend_advance_example.jpg)    

Node options:   
![image](image/image_blend_advance_node.jpg)    
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
* [note](#notes)

### <a id="table1">CropByMask</a>
Crop the image according to the mask range, and set the size of the surrounding borders to be retained.
This node can be used in conjunction with the [RestoreCropBox](#RestoreCropBox) and [ImageScaleRestore](#ImageScaleRestore) nodes to crop and modify upscale parts of image, and then paste them back in place.
![image](image/corp_by_mask_example.jpg)    

Node options:   
![image](image/corp_by_mask_node.jpg)    
* image<sup>5</sup>: The input image.
* mask_for_crop<sup>5</sup>: Mask of the image, it will automatically be cut according to the mask range.
* invert_mask: Whether to reverse the mask.
* detect: Detection method, ```min_bounding_rect``` is the minimum bounding rectangle of block shape, ```max_inscribed_rect``` is the maximum inscribed rectangle of block shape, and ```mask-area``` is the effective area for masking pixels.
* top_reserve: Cut the top to preserve size.
* bottom_reserve: Cut the bottom to preserve size.
* left_reserve: Cut the left to preserve size.
* right_reserve: Cut the right to preserve size.
* [note](#notes)

Output:
* croped_image: The image after crop.
* croped_mask: The mask after crop.
* crop_box: The trimmed box data is used when restoring the RestoreCropBox node.
* box_preview: Preview image of cutting position, red represents the detected range, and green represents the cutting range after adding the reserved border.

### <a id="table1">CropByMaskV2</a>
The V2 upgraded version of CropByMask. Supports crop_box input, making it easy to cut layers of the same size.

The following changes have been made based on CropByMask:
![image](image/corp_by_mask_v2_node.jpg)    
* The input ```mask_for_crop``` reanme to ```mask```。
* Add optional inputs to the ```crop_box```. If there are inputs here, mask detection will be ignored and this data will be directly used for cropping.
* Add the option ```round_to_multiple``` to round the trimming edge length multiple. For example, setting it to 8 will force the width and height to be multiples of 8.

### <a id="table1">RestoreCropBox</a>
Restore the cropped image to the original image by [CropByMask](#CropByMask).

Node options:   
![image](image/restore_crop_box_node.jpg)    
* background_image: The original image before cutting.
* croped_image<sup>5</sup>: The cropped image. If the middle is enlarged, the size needs to be restored before restoration.
* croped_mask<sup>5</sup>: The cut mask.
* crop_box: Box data during cutting.
* invert_mask: Whether to reverse the mask.
* [note](#notes)

### <a id="table1">CropBoxResolve</a>
Parsing the ```corp_box```  to ```x``` , ```y``` , ```width``` , ```height``` .
![image](image/corp_box_resolve_node.jpg)    

### <a id="table1">ImageScaleRestore</a>
Image scaling. when this node is used in pairs, the image can be automatically restored to its original size on the second node.
![image](image/image_scale_restore_example.jpg)    

Node options:   
![image](image/image_scale_restore_node.jpg)    
* image<sup>5</sup>: The input image.
* mask<sup>2,5</sup>: Mask of image.
* original_size: Optional input, used to restore the image to its original size.
* scale: Scale ratio. when the original_size have input, or scale_ by_longest_side is set to True, this setting will be ignored.
* scale_by_longest_side: Allow scaling by long edge size.
* longest_side: When the scale_by_longest_side is set to True, this will be used this value to the long edge of the image. when the original_size have input, this setting will be ignored.

Outputs:
* image: The scaled image.
* mask: If have mask input, the scaled mask will be output.
* original_size: The original size data of the image is used for subsequent node recovery.
* width: The output image's width.
* height: The output image's height.

### <a id="table1">ImageScaleRestoreV2</a>
The V2 upgraded version of ImageScaleRestore.

Node options:   
![image](image/image_scale_restore_v2_node.jpg)    
The following changes have been made based on ImageScaleRestore:
* scale_by: Allow scaling by specified dimensions for long, short, width, height, or total pixels. When this option is set to by_scale, use the scale value, and for other options, use the scale_by_length value.
* scale_by_length: The value here is used as ```scale_by``` to specify the length of the edge.

### <a id="table1">ImageMaskScaleAs</a>
Scale the image or mask to the size of the reference image (or reference mask).
![image](image/image_mask_scale_as_example.jpg)    

Node options:
![image](image/image_mask_scale_as_node.jpg)    
* scale_as<sup>*</sup>: Reference size. It can be an image or a mask.
* image: Image to be scaled. this option is optional input. if there is no input, a black image will be output.
* mask: Mask to be scaled. this option is optional input. if there is no input, a black mask will be output.
* fit: Scale aspect ratio mode. when the width to height ratio of the original image does not match the scaled size, there are three modes to choose from, 
The _letterbox_ mode retains the complete frame and fills in the blank spaces with black; 
The _crop_ mode retains the complete short edge, and any excess of the long edge will be cut off;
The _fill_ mode does not maintain frame ratio and fills the screen with width and height.
* method: Scaling sampling methods, including lanczos, bicubic, hamming, bilinear, box, and nearest.

<sup>*</sup>Only limited to input images and masks. forcing the integration of other types of inputs will result in node errors.

Outputs:
* image: If there is an image input, the scaled image will be output.
* mask: If there is a mask input, the scaled mask will be output.
* original_size: The original size data of the image is used for subsequent node recovery.
* width: The output image's width.
* height: The output image's height.

### <a id="table1">ImageScaleByAspectRatio</a>
Scale the image or mask by aspect ratio. the scaled size can be rounded to a multiple of 8 or 16, and can be scaled to the long side size.
![image](image/image_scale_by_aspect_ratio_example.jpg)    

Node options:   
![image](image/image_scale_by_aspect_ratio_node.jpg)    
* aspect_ratio: Here are several common frame ratios provided. alternatively, you can choose "original" to keep original ratio or customize the ratio using "custom".
* proportional_width: Proportional width. if the aspect ratio option is not "custom", this setting will be ignored.
* proportional_height: Proportional height. if the aspect ratio option is not "custom", this setting will be ignored.
* fit: Scale aspect ratio mode. when the width to height ratio of the original image does not match the scaled size, there are three modes to choose from, 
The _letterbox_ mode retains the complete frame and fills in the blank spaces with black; 
The _crop_ mode retains the complete short edge, and any excess of the long edge will be cut off;
The _fill_ mode does not maintain frame ratio and fills the screen with width and height.
* method: Scaling sampling methods, including lanczos, bicubic, hamming, bilinear, box, and nearest.
* round_to_multiple: Round multiples. for example, setting it to 8 will force the width and height to be multiples of 8.
* scale_by_longest_side: Allow scaling by long edge size.
* longest_side: When the scale_by_longest_side is set to True, this will be used this value to the long edge of the image. when the original_size have input, this setting will be ignored.


Outputs:
* image: If have image input, the scaled image will be output.
* mask: If have mask input, the scaled mask will be output.
* original_size: The original size data of the image is used for subsequent node recovery.
* width: The output image's width.
* height: The output image's height.

### <a id="table1">ImageScaleByAspectRatioV2</a>
V2 Upgraded Version of ImageScaleByAspectRatio

Node options:   
![image](image/image_scale_by_aspect_ratio_v2_node.jpg)    
The following changes have been made based on ImageScaleByAspectRatio:
* scale_to_side: Allow scaling by specified dimensions for long, short, width, height, or total pixels.
* scale_to_length: The numerical value here serves as the length of the specified edge or the total pixels (kilo pixels) for scale_to_side.
* background_color<sup>4</sup>: The color of the background.


### <a id="table1">QWenImage2Prompt</a>
Inference the prompts based on the image. this node is repackage of the [ComfyUI_VLM_nodes](https://github.com/gokayfem/ComfyUI_VLM_nodes)'s ```UForm-Gen2 Qwen Node```,  thanks to the original author.
Download model files from [huggingface](https://huggingface.co/unum-cloud/uform-gen2-qwen-500m) or [Baidu Netdisk](https://pan.baidu.com/s/1ztnVX_Sh800xsWZhMDe-Ww?pwd=esyt) to ```ComfyUI/models/LLavacheckpoints/files_for_uform_gen2_qwen``` folder.

![image](image/qwen_image2prompt_example.jpg)    

Node Options:   
* question: Prompt of UForm-Gen-QWen model.



### <a id="table1">PromptTagger</a>
Inference the prompts based on the image. it can replace key word for the prompt. This node currently uses Google Gemini API as the backend service. Please ensure that the network environment can use Gemini normally.
Please apply for your API key on [Google AI Studio](https://makersuite.google.com/app/apikey),  And fill it in ```api_key.ini```, this file is located in the root directory of the plug-in, and the default name is ```api_key.ini.example```. to use this file for the first time, you need to change the file suffix to ```.ini```. Open it using text editing software, fill in your API key after ```google_api_key=``` and save it.
![image](image/prompt_tagger_example.jpg)    

Node options:   
![image](image/prompt_tagger_node.jpg)    

* api: The Api used. At present, there are two options "gemini-1. 5-flash" and "google-gemini".
* token_limit: The maximum token limit for generating prompt words.
* exclude_word: Keywords that need to be excluded.
* replace_with_word: That word will replace the exclude_word.


### <a id="table1">PromptEmbellish</a>
Enter simple prompt words, output polished prompt words, and support inputting images as references, and support Chinese input. This node currently uses Google Gemini API as the backend service. Please ensure that the network environment can use Gemini normally.
Please apply for your API key on [Google AI Studio](https://makersuite.google.com/app/apikey),  And fill it in ```api_key.ini```, this file is located in the root directory of the plug-in, and the default name is ```api_key.ini.example```. to use this file for the first time, you need to change the file suffix to ```.ini```. Open it using text editing software, fill in your API key after ```google_api_key=``` and save it.
![image](image/prompt_embellish_example.jpg)    

Node options:   
![image](image/prompt_embellish_node.jpg)    

* image: Optional, input image as a reference for prompt words.
* api: The Api used. At present, there are two options "gemini-1. 5-flash" and "google-gemini".
* token_limit: The maximum token limit for generating prompt words.
* discribe: Enter a simple description here. supports Chinese text input.

### <a id="table1">Florence2Image2Prompt</a>
Use the Florence 2 model to infer prompt words. The code for this node section is from[yiwangsimple/florence_dw](https://github.com/yiwangsimple/florence_dw), thanks to the original author.
*When using it for the first time, the model will be automatically downloaded.
![image](image/florence2_image2prompt_example.jpg) 

Node Options:
![image](image/florence2_image2prompt_node.jpg)
* florence2_model: Florence2 model input.
* image: Image input.
* task: Select the task for florence2.
* text_input: Text input for florence2.
* max_new_tokens: The maximum number of tokens for generating text.
* num_beams: The number of beam searches that generate text.
* do_sample: Whether to use text generated sampling.
* fill_mask: Whether to use text marker mask filling.

### <a id="table1">ImageShift</a>
Shift the image. this node supports the output of displacement seam masks, making it convenient to create continuous textures.
![image](image/image_shift_example.jpg)    

Node options:   
![image](image/image_shift_node.jpg)    
* image<sup>5</sup>: The input image.
* mask<sup>2,5</sup>: The mask of image.
* shift_x: Horizontal distance of shift.
* shift_y: Vertical distance of shift.
* cyclic: Is the part of displacement that is out of bounds cyclic.
* background_color<sup>4</sup>: Background color. if cyclic is set to False, the setting here will be used as the background color.
* border_mask_width: Border mask width.
* border_mask_blur: Border mask blur.
* [note](#notes)

### <a id="table1">ImageBlend</a>
A simple node for composit layer image and background image, multiple blend modes are available for option, and transparency can be set.
![image](image/image_blend_example.jpg)    

Node options:
![image](image/image_blend_node.jpg)    
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1,2</sup>: Mask for layer_image.
* invert_mask: Whether to reverse the mask.
* blend_mode<sup>3</sup>: Blending mode.
* opacity: Opacity of blend.
* [note](#notes)

### <a id="table1">ImageOpacity</a>
Adjust image opacity
![image](image/image_opacity_example.jpg)    

Node option:   
* image<sup>5</sup>: Image input, supporting RGB and RGBA. if is RGB, the alpha channel of the entire image will be automatically added.
* mask<sup>2,5</sup> : Mask input.
* invert_mask: Whether to reverse the mask.
* opacity: Opacity of image.
* [note](#notes)

### <a id="table1">ColorPicker</a>
Modify web extensions from [mtb nodes](https://github.com/melMass/comfy_mtb). Select colors on the color palette and output RGB values, thanks to the original author.
![image](image/color_picker.jpg)    

Node options:
* mode： The output format is available in hexadecimal (HEX) and decimal (DEC).  

Output type: 
* value: String format.

### <a id="table1">RGBValue</a>
Output the color value as a single R, G, B three decimal values. Supports HEX and DEC formats for ColorPicker node output.

![image](image/RGB_value_example.jpg)    

Node Options:
* color_value: Supports hexadecimal (HEX) or decimal (DEC) color values and should be of string or tuple type. Forcing in other types will result in an error.

### <a id="table1">HSVValue</a>
Output color values as individual decimal values of H, S, and V (maximum value of 255). Supports HEX and DEC formats for ColorPicker node output.
![image](image/hsv_value_node.jpg)    

Node Options:
* color_value: Supports hexadecimal (HEX) or decimal (DEC) color values and should be of string or tuple type. Forcing in other types will result in an error.

### <a id="table1">GetColorTone</a>
Obtain the main color or average color from the image and output RGB values.
![image](image/get_color_tone_example.jpg)    

Node options:
![image](image/get_color_tone_node.jpg)    
* mode： There are two modes to choose from, with the main color and average color.

Output type:
* RGB color in HEX: The RGB color described by hexadecimal RGB format, like '#FA3D86'.
* HSV color in list: The HSV color described by python's list data format.

### <a id="table1">GetColorToneV2</a>
V2 upgrade of GetColorTone. You can specify the dominant or average color to get the body or background.
![image](image/get_color_tone_v2_example.jpg)    

The following changes have been made on the basis of GetColorTong:
![image](image/get_color_tone_v2_node.jpg)    
* color_of: Provides 4 options, mask, entire, background, and subject, to select the color of the mask area, entire picture, background, or subject, respectively.
* remove_background_method: There are two methods of background recognition: BiRefNet and RMBG V1.4.
* invert_mask: Whether to reverse the mask.
* mask_grow: Mask expansion. For subject, a larger value brings the obtained color closer to the color at the center of the body.

Output:
* image: Solid color picture output, the size is the same as the input picture.
* mask: Mask output.

### <a id="table1">ExtendCanvas</a>
Extend the canvas
![image](image/extend_canvas_example.jpg)    

Node options:
![image](image/extend_canvas_node.jpg)    
* invert_mask: Whether to reverse the mask.
* top: Top extension value.
* bottom: Bottom extension value.
* left: Left extension value.
* right: Right extension value.
* color; Color of canvas.

### <a id="table1">ExtendCanvasV2</a>
V2 upgrade to ExtendCanvas.

Based on ExtendCanvas, color is modified to be a string type, and it supports external ```ColorPicker``` input.
![image](image/extend_canvas_v2_node.jpg)    

### XY to <a id="table1">Percent</a>
![image](image/xy2percent_example.jpg)    
Convert absolute coordinates to percentage coordinates.

![image](image/xy2percent_node.jpg)    
Node options:
* x: Value of X.
* y: Value of Y.

### <a id="table1">LayerImageTransform</a>
![image](image/layer_image_transform_example.jpg)    
This node is used to transform layer_image separately, which can change size, rotation, aspect ratio, and mirror flip without changing the image size.

![image](image/layer_image_transform_node.jpg)    
Node options:
* x: Value of X.
* y: Value of Y.
* mirror: Mirror flipping. Provide two flipping modes, horizontal flipping and vertical flipping.
* scale: Layer magnification, 1.0 represents the original size.
* aspect_ratio: Layer aspect ratio. 1.0 is the original ratio, a value greater than this indicates elongation, and a value less than this indicates flattening.
* rotate: Layer rotation degree.
* Sampling methods for layer enlargement and rotation, including lanczos, bicubic, hamming, bilinear, box and nearest. Different sampling methods can affect the image quality and processing time of the synthesized image.
* anti_aliasing: Anti aliasing, ranging from 0 to 16, the larger the value, the less obvious the aliasing. An excessively high value will significantly reduce the processing speed of the node.


### <a id="table1">LayerMaskTransform</a>
Similar to LayerImageTransform node, this node is used to transform the layer_mask separately, which can scale, rotate, change aspect ratio, and mirror flip without changing the mask size.

![image](image/layer_mask_transform_node.jpg)    
Node options:
* x: Value of X.
* y: Value of Y.
* mirror: Mirror flipping. Provide two flipping modes, horizontal flipping and vertical flipping.
* scale: Layer magnification, 1.0 represents the original size.
* aspect_ratio: Layer aspect ratio. 1.0 is the original ratio, a value greater than this indicates elongation, and a value less than this indicates flattening.
* rotate: Layer rotation degree.
* Sampling methods for layer enlargement and rotation, including lanczos, bicubic, hamming, bilinear, box and nearest. Different sampling methods can affect the image quality and processing time of the synthesized image.
* anti_aliasing: Anti aliasing, ranging from 0 to 16, the larger the value, the less obvious the aliasing. An excessively high value will significantly reduce the processing speed of the node.


### <a id="table1">ColorImage</a>
![image](image/color_image_example.jpg)    
Generate an image of a specified color and size.

![image](image/color_image_node.jpg)    
Node options:
* width: Width of the image.
* height: Height of the image.
* color<sup>4</sup>: Color of the image.

### <a id="table1">ColorImageV2</a>
The V2 upgraded version of ColorImage.

![image](image/color_image_v2_node.jpg)    
The following changes have been made based on ColorImage:
* size_as<sup>*</sup>: Input image or mask here to generate image according to its size. Note that this input takes priority over other size settings.
* size<sup>**</sup>: Size preset. the preset can be customized by the user. if have size_as input, this option will be ignored.
* custom_width: Image width. it valid when size is set to "custom". if have size_as input, this option will be ignored.
* custom_height: Image height. it valid when size is set to "custom". if have size_as input, this option will be ignored.

<sup>*</sup>Only limited to input images and masks. forcing the integration of other types of inputs will result in node errors.
<sup>**</sup>The preset size is defined in ```custom_size.ini```, this file is located in the root directory of the plug-in, and the default name is ```custom_size.ini.example```. to use this file for the first time, you need to change the file suffix to ```.ini```. Open with text editing software. Each row represents a size, with the first value being width and the second being height, separated by a lowercase "x" in the middle. To avoid errors, please do not enter extra characters.


### <a id="table1">GradientImage</a>
![image](image/gradient_image_example.jpg)    
Generate an image with a specified size and color gradient.

![image](image/gradient_image_node.jpg)    
Node options:
* width: Width of the image.
* height: Height of the image.
* angle: Angle of gradient.
* start_color<sup>4</sup>: Color of the begging.
* end_color<sup>4</sup>: Color of the ending.

### <a id="table1">GradientImageV2</a>
The V2 upgraded version of GradientImage.

![image](image/gradient_image_node_v2.jpg)    
The following changes have been made based on GradientImage:
* size_as<sup>*</sup>: Input image or mask here to generate image according to its size. Note that this input takes priority over other size settings.
* size<sup>**</sup>: Size preset. the preset can be customized by the user. if have size_as input, this option will be ignored.
* custom_width: Image width. it valid when size is set to "custom". if have size_as input, this option will be ignored.
* custom_height: Image height. it valid when size is set to "custom". if have size_as input, this option will be ignored.

<sup>*</sup>Only limited to input images and masks. forcing the integration of other types of inputs will result in node errors.
<sup>**</sup>The preset size is defined in ```custom_size.ini```, this file is located in the root directory of the plug-in, and the default name is ```custom_size.ini.example```. to use this file for the first time, you need to change the file suffix to ```.ini```. Open with text editing software. Each row represents a size, with the first value being width and the second being height, separated by a lowercase "x" in the middle. To avoid errors, please do not enter extra characters.

### <a id="table1">ImageRewardFilter</a>
![image](image/image_reward_filter_example.jpg)    
Rating bulk pictures and outputting top-ranked pictures. it used [ImageReward] (https://github.com/THUDM/ImageReward) for image scoring, thanks to the original authors.

![image](image/image_reward_filter_node.jpg)    
Node options:
* prompt: Optional input. Entering prompt here will be used as a basis to determine how well it matches the picture.
* output_nun: Number of pictures outputted. This value should be less than the picture batch.

Outputs：
* images: Bulk pictures output from high to low in order of rating.
* obsolete_images: Knockout pictures. Also output in order of rating from high to low.

### <a id="table1">SimpleTextImage</a>
![image](image/simple_text_image_example.jpg)    
Generate simple typesetting images and masks from text. This node references some of the functionalities and code of [ZHO-ZHO-ZHO/ComfyUI-Text_Image-Composite](https://github.com/ZHO-ZHO-ZHO/ComfyUI-Text_Image-Composite), thanks to the original author.

![image](image/simple_text_image_node.jpg)    
Node options:
* size_as<sup>*</sup>: The input image or mask here will generate the output image and mask according to their size. this input takes priority over the width and height below.
* font_file<sup>**</sup>: Here is a list of available font files in the font folder, and the selected font files will be used to generate images.
* align: Alignment options. There are three options: center, left, and right.
* char_per_line: The number of characters per line, any excess will be automatically wrapped.
* leading: The leading space.
* font_size: The size of font.
* text_color: The color of text.
* stroke_width: The width of stroke.
* stroke_color: The color of stroke.
* x_offset: The horizontal offset of the text position.
* y_offset: The vertical offset of the text position.
* width: Width of the image. If there is a size_as input, this setting will be ignored.
* height: Height of the image. If there is a size_as input, this setting will be ignored.


<sup>*</sup>Only limited to input image and mask. forcing the integration of other types of inputs will result in node errors.

<sup>**</sup>The font folder is defined in ```resource_dir.ini```, this file is located in the root directory of the plug-in, and the default name is ```resource_dir.ini.example```. to use this file for the first time, you need to change the file suffix to ```.ini```.
Open the text editing software and find the line starting with "FONT_dir=", after "=", enter the custom folder path name. all font files in this folder will be collected and displayed in the node list during ComfyUI initialization.
If the folder set in ini is invalid, the font folder that comes with the plugin will be enabled.



### <a id="table1">TextImage</a>
![image](image/text_image_example.jpg)    
Generate images and masks from text. support for adjusting the spacing between words and lines, horizontal and vertical adjustments, it can set random changes in each character, including size and position.

![image](image/text_image_node.jpg)    
Node options:
* size_as<sup>*</sup>: The input image or mask here will generate the output image and mask according to their size. this input takes priority over the width and height below.
* font_file<sup>**</sup>: Here is a list of available font files in the font folder, and the selected font files will be used to generate images.
* spacing: Word spacing.this value is in pixels.
* leading: Row leading.this value is in pixels.
* horizontal_border: Side margin. If the text is horizontal, it is the left margin, and if it is vertical, it is the right margin. this value is represents a percentage, for example, 50 indicates that the starting point is located in the center on both sides.
* vertical_border: Top margin. this value is represents a percentage, for example, 10 indicates that the starting point is located 10% away from the top.
* scale: The overall size of the text. the initial size of text is automatically calculated based on the screen size and text content, with the longest row or column by default adapting to the image width or height. adjusting the value here will scale the text as a whole. this value is represents a percentage, for example, 60 represents scaling to 60%.
* variation_range: The range of random changes in characters. when this value is greater than 0, the character will undergo random changes in size and position, and the larger the value, the greater the magnitude of the change.
* variation_seed: The seed for randomly. fix this value to individual characters changes generated each time will not change.
* layout: Text layout. there are horizontal and vertical options to choose from.
* width: Width of the image. If there is a size_as input, this setting will be ignored.
* height: Height of the image. If there is a size_as input, this setting will be ignored.
* text_color: The color of text.
* background_color<sup>4</sup>: The color of background.

<sup>*</sup>Only limited to input image and mask. forcing the integration of other types of inputs will result in node errors.

<sup>**</sup>The font folder is defined in ```resource_dir.ini```, this file is located in the root directory of the plug-in, and the default name is ```resource_dir.ini.example```. to use this file for the first time, you need to change the file suffix to ```.ini```.
Open the text editing software and find the line starting with "FONT_dir=", after "=", enter the custom folder path name. all font files in this folder will be collected and displayed in the node list during ComfyUI initialization.
If the folder set in ini is invalid, the font folder that comes with the plugin will be enabled.


### <a id="table1">LaMa</a>
![image](image/lama_example.jpg)    
Erase objects from the image based on the mask. this node is repackage of [IOPaint](https://www.iopaint.com), powered by state-of-the-art AI models, thanks to the original author.    
It is have [LaMa](https://github.com/advimman/lama), [LDM](https://github.com/CompVis/latent-diffusion), [ZITS](https://github.com/DQiaole/ZITS_inpainting),[MAT](https://github.com/fenglinglwb/MAT),  [FcF](https://github.com/SHI-Labs/FcF-Inpainting), [Manga](https://github.com/msxie92/MangaInpainting) models and the SPREAD method to erase. Please refer to the original link for the introduction of each model.    
Please download the model files from [lama models(Baidu Netdisk)](https://pan.baidu.com/s/1LllR9TJHP1G9uEwWT3Mvkg?pwd=tvzv) or [lama models(Google Drive)](https://drive.google.com/drive/folders/1Aq0a4sybb3SRxi7j1e1_ZbBRjaWDdP9e?usp=sharing) to ```ComfyUI/models/lama``` folder.    

Node optons:
![image](image/lama_node.jpg)    
* lama_model: Choose a model or method.
* device: After correctly installing Torch and Nvidia CUDA drivers, using cuda will significantly improve running speed.
* invert_mask: Whether to reverse the mask.
* grow: Positive values expand outward, while negative values contract inward.
* blur: Blur the edge.


### <a id="table1">ImageChannelSplit</a>
![image](image/image_channel_split_example.jpg)    
Split the image channel into individual images.

Node options:
![image](image/image_channel_split_node.jpg)    
* mode: Channel mode, include RGBA, YCbCr, LAB adn HSV.

### <a id="table1">ImageChannelMerge</a>
![image](image/image_channel_merge_example.jpg)    
Merge each channel image into one image.

Node options:
![image](image/image_channel_merge_node.jpg)    
* mode: Channel mode, include RGBA, YCbCr, LAB adn HSV.

### <a id="table1">ImageRemoveAlpha</a>
![image](image/image_remove_alpha_example.jpg)    
Remove the alpha channel from the image and convert it to RGB mode. you can choose to fill the background and set the background color.

Node options:
![image](image/image_remove_alpha_node.jpg)    
* RGBA_image: The input image supports RGBA or RGB modes.
* mask: Optional input mask. If there is an input mask, it will be used first, ignoring the alpha that comes with RGBA_image.
* fill_background: Whether to fill the background.
* background_color<sup>4</sup>: Color of background.


### <a id="table1">ImageCombineAlpha</a>
![image](image/image_combine_alpha_node.jpg)    
Merge the image and mask into an RGBA mode image containing an alpha channel.

### <a id="table1">ImageAutoCrop</a>
![image](image/image_auto_crop_example.jpg)    
Automatically cutout and crop the image according to the mask. it can specify the background color, aspect ratio, and size for output image. this node is designed to generate the image materials for training models.   
*Please refer to the model installation methods for [SegmentAnythingUltra](#SegmentAnythingUltra) and [RemBgUltra](#RemBgUltra).  


Node options:
![image](image/image_auto_crop_node.jpg)    
* background_color<sup>4</sup>: The background color.
* aspect_ratio: Here are several common frame ratios provided. alternatively, you can choose "original" to keep original ratio or customize the ratio using "custom".
* proportional_width: Proportional width. if the aspect ratio option is not "custom", this setting will be ignored.
* proportional_height: Proportional height. if the aspect ratio option is not "custom", this setting will be ignored.
* scale_by_longest_side: Allow scaling by long edge size.
* longest_side: When the scale_by_longest_side is set to True, this will be used this value to the long edge of the image. when the original_size have input, this setting will be ignored.
* detect: Detection method, min_bounding_rect is the minimum bounding rectangle, max_inscribed_rect is the maximum inscribed rectangle.
* border_reserve: Keep the border. expand the cutting range beyond the detected mask body area.
* ultra_detail_range: Mask edge ultra fine processing range, 0 is not processed, which can save generation time.
* matting_method: The method of generate masks. There are two methods available: Segment Anything and RMBG 1.4. RMBG 1.4 runs faster.
* sam_model: Select the SAM model used by Segment Anything here.
* grounding_dino_model: Select the Grounding_Dino model used by Segment Anything here.
* sam_threshold: The threshold for Segment Anything.
* sam_prompt: The prompt for Segment Anything.

Output:
cropped_image: Crop and replace the background image.
box_preview: Crop position preview.
cropped_mask: Cropped mask.


### <a id="table1">ImageAutoCropV2</a>

The V2 upgrad version of ```ImageAutoCrop```, it has made the following changes based on the previous version:
![image](image/image_auto_crop_v2_node.jpg)    

* Add optional input for mask. when there is a mask input, use that input directly to skip the built-in mask generation.
* Add ```fill_background```. When set to False, the background will not be processed and any parts beyond the frame will not be included in the output range.
* ```aspect_ratio``` adds the ```original``` option.
* scale_by: Allow scaling by specified dimensions for longest, shortest, width, or height.
* scale_by_length: The value here is used as ```scale_by``` to specify the length of the edge.


### <a id="table1">HLFrequencyDetailRestore</a>
Using low frequency filtering and retaining high frequency to recover image details. Compared to [kijai's DetailTransfer](https://github.com/kijai/ComfyUI-IC-Light), this node is better integrated with the environment while retaining details.
![image](image/hl_frequency_detail_restore_example.jpg)    

Node Options:  
![image](image/hl_frequency_detail_restore_node.jpg)    
* image: Background image input.
* detail_image: Detail image input.
* mask: Optional input, if there is a mask input, only the details of the mask part are restored.
* keep_high_freq: Reserved range of high frequency parts. The larger the value, the richer the retained high-frequency details.
* erase_low_freq: The range of low frequency parts of the erasure. The larger the value, the more the low frequency range of the erasure.
* mask_blur: Mask edge blur. Valid only if there is masked input.

### <a id="table1">GetImageSize</a>
![image](image/get_image_size_node.jpg)    
Obtain the width and height of the image.

Output:
* width: The width of image.
* height: The height of image.
* original_size: The original size data of the image is used for subsequent node recovery.

### <a id="table1">ImageHub</a>
Switch output from multiple input images and masks, supporting 9 sets of inputs. All input items are optional. if there is only image or mask in a set of input, the missing item will be output as None.
![image](image/image_hub_example.jpg)    

Node options:  
![image](image/image_hub_node.jpg)    
* output: Switch output. the value is the corresponding input group. when the ```random-output``` option is True, this setting will be ignored.
* random_output: When this is true, the ```output``` setting will be ignored and a random set will be output among all valid inputs.


### <a id="table1">BatchSelector</a>
Retrieve specified images or masks from batch images or masks.
![image](image/batch_selector_example.jpg)    

Node Options:  
![image](image/batch_selector_node.jpg)    
* images: Batch images input. This input is optional.
* masks: Batch masks input. This input is optional.
* select: Select the output image or mask at the batch index value, where 0 is the first image. Multiple values can be entered, separated by any non numeric character, including but not limited to commas, periods, semicolons, spaces or letters, and even Chinese characters.
Note: If the value exceeds the batch size, the last image will be output. If there is no corresponding input, an empty 64x64 image or a 64x64 black mask will be output.


### <a id="table1">TextJoin</a>
![image](image/text_join_example.jpg)    
Combine multiple paragraphs of text into one.

### <a id="table1">PrintInfo</a>
![image](image/print_info_node.jpg)    
Used to provide assistance for workflow debugging. When running, the properties of any object connected to this node will be printed to the console.

This node allows any type of input.


### <a id="table1">TextBox</a>
![image](image/text_box_node.jpg)    
Output a string.

### <a id="table1">String</a>
![image](image/string_node.jpg)    
Output a string. same as TextBox.

### <a id="table1">Integer</a>
![image](image/integer_node.jpg)    
Output a integer value.

### <a id="table1">Float</a>
![image](image/float_node.jpg)    
Output a floating-point value with a precision of 5 decimal places.

### <a id="table1">Boolean</a>
![image](image/boolean_node.jpg)    
Output a boolean value.

### <a id="table1">NumberCalculator</a>
![image](image/number_calculator_node.jpg)    
Performs mathematical operations on two numeric values and outputs integer and floating point results<sup>*</sup>. Supported operations include```+```, ```-```, ```*```, ```/```, ```**```, ```//```, ```%```.

<sup>*</sup>  The input only supports boolean, integer, and floating point numbers, forcing in other data will result in error.

### <a id="table1">NumberCalculatorV2</a>
![image](image/number_calculator_v2_node.jpg)  
The upgraded version of NumberCalculator has added numerical inputs within nodes and square root operations. The square root operation option is ```nth_root```
Note: The input takes priority, and when there is input, the values within the node will be invalid.

### <a id="table1">BooleanOperator</a>
![image](image/boolean_operator_node.jpg)    
Perform a Boolean operation on two numeric values and output the result<sup>*</sup>. Supported operations include```==```, ```!=```, ```and```, ```or```, ```xor```, ```not```, ```min```, ```max```.

<sup>*</sup>  The input only supports boolean, integer, and floating point numbers, forcing in other data will result in error. The ```and``` operation between the values outputs a larger number, and the ```or``` operation outputs a smaller number.


### <a id="table1">BooleanOperatorV2</a>
![image](image/boolean_operator_v2_node.jpg)  
The upgraded version of Boolean Operator has added numerical inputs within nodes and added judgments for greater than, less than, greater than or equal to, and less than or equal to.
Note: The input takes priority, and when there is input, the values within the node will be invalid.


### <a id="table1">StringCondition</a>
![image](image/string_condition_example.jpg)    
Determine whether the text contains or does not contain substrings, and output a Boolean value.

Node Options:    
![image](image/string_condition_node.jpg)  
* text: Input text.
* condition: Judgment conditions. ```include``` determines whether it contains a substring, and ```exclude``` determines whether it does not.
* sub_string: Substring.

### <a id="table1">CheckMask</a>
Check if the mask contains enough valid areas and output a Boolean value.

Node Options:    
![image](image/check_mask_node.jpg)    
* white_point: The white point threshold used to determine whether the mask is valid is considered valid if it exceeds this value.
* area_percent: The percentage of effective areas. If the proportion of effective areas exceeds this value, output True.


### <a id="table1">If</a>
![image](image/if_example.jpg)    
Switches output based on Boolean conditional input. It can be used for any type of data switching, including but not limited to numeric values, strings, pictures, masks, models, latent, pipe pipelines, etc.

Node Options:    
![image](image/if_node.jpg)    
* if_condition: Conditional input. Boolean, integer, floating point, and string inputs are supported. When entering a value, 0 is judged to be False; When a string is entered, an empty string is judged as Flase.
* when_True: This item is output when the condition is True.
* when_False: This item is output when the condition is False.

### <a id="table1">SwitchCase</a>
![image](image/switch_case_example.jpg)    
Switches the output based on the matching string. It can be used for any type of data switching, including but not limited to numeric values, strings, pictures, masks, models, latent, pipe pipelines, etc. Supports up to 3 sets of case switches.
Compare case to ```switch_condition``` , if the same, output the corresponding input. If there are the same cases, the output is prioritized in order. If there is no matching case, the default input is output. 
Note that the string is case sensitive and Chinese and English full-width and half-width.

Node Options:    
![image](image/switch_case_node.jpg)    
* input_default: Input entry for default output. This input is required.
* input_1: Input entry used to match ```case_1```. This input is optional.
* input_2: Input entry used to match ```case_2```. This input is optional.
* input_3: Input entry used to match ```case_3```. This input is optional.
* switch_condition: String used to judge with case.
* case_1: case_1 string.
* case_2: case_2 string.
* case_3: case_3 string.

### <a id="table1">QueueStop</a>
![image](image/queue_stop_example.jpg)    
Stop the current queue. When executed at this node, the queue will stop. The workflow diagram above illustrates that if the image is larger than 1Mega pixels, the queue will stop executing.

Node Options:    
![image](image/queue_stop_node.jpg)    
* mode: Stop mode. If you choose ```stop```, it will be determined whether to stop based on the input conditions. If you choose ```continue```, ignore the condition to continue executing the queue.
* stop: If true, the queue will stop. If false, the queue will continue to execute.

### <a id="table1">PurgeVRAM</a>
![image](image/purge_vram_example.jpg)    
Clean up GPU VRAM and system RAM. any type of input can be accessed, and when executed to this node, the VRAM and garbage objects in the RAM will be cleaned up. Usually placed after the node where the inference task is completed, such as the VAE Decode node.

Node Options:  
* purge_cache: Clean up cache。
* purge_models: Unload all loaded models。



### <a id="table1">SaveImagePlus</a>
![image](image/saveimage_plus_example.jpg)  
Enhanced save image node. You can customize the directory where the picture is saved, add a timestamp to the file name, select the save format, set the image compression rate, set whether to save the workflow, and optionally add invisible watermarks to the picture. (Add information in a way that is invisible to the naked eye, and use the ```ShowBlindWaterMark``` node to decode the watermark). Optionally output the json file of the workflow.

Node Options:
![image](image/saveimage_plus_node.jpg)    
* iamge: The input image.
* custom_path<sup>*</sup>: User-defined directory, enter the directory name in the correct format. If empty, it is saved in the default output directory of ComfyUI.
* filename_prefix<sup>*</sup>: The prefix of file name.
* timestamp: Timestamp the file name, opting for date, time to seconds, and time to milliseconds.
* format: The format of image save. Currently available in ```png``` and ```jpg```. Note that only png format is supported for RGBA mode pictures.
* quality: Image quality, the value range 10-100, the higher the value, the better the picture quality, the volume of the file also correspondingly increases.
* meta_data: Whether to save metadata to png file, that is workflow information. Set this to false if you do not want the workflow to be leaked.
* blind_watermark: The text entered here (does not support multilingualism) will be converted into a QR code and saved as an invisible watermark. Use ```ShowBlindWaterMark``` node can decode watermarks. Note that pictures with watermarks are recommended to be saved in png format, and lower-quality jpg format will cause watermark information to be lost.
* save_workflow_as_json: Whether the output workflow is a json file at the same time (the output json is in the same directory as the picture).
* preview: Preview switch.

<sup>*</sup> Enter```%date``` for the current date (YY-mm-dd) and ```%time``` for the current time (HH-MM-SS). You can enter ```/``` for subdirectories. For example, ```%date/name_%tiem``` will output the image to the ```YY-mm-dd``` folder, with ```name_HH-MM-SS``` as the file name prefix.



### <a id="table1">AddBlindWaterMark</a>
![image](image/watermark_example.jpg)    
Add an invisible watermark to a picture. Add the watermark image in a way that is invisible to the naked eye, and use the ```ShowBlindWaterMark``` node to decode the watermark.

Node Options:
![image](image/add_blind_watermark_node.jpg)    
* iamge: The input image.
* watermark_image: Watermark image. The image entered here will automatically be converted to a square black and white image as a watermark. It is recommended to use a QR code as a watermark.


### <a id="table1">ShowBlindWaterMark</a>
Decoding the invisible watermark added to the ```AddBlindWaterMark``` and ```SaveImagePlus``` nodes.
![image](image/show_blind_watermark_node.jpg)    


### <a id="table1">CreateQRCode</a>
Generate a square QR code picture.

Node Options:  
![image](image/create_qrcode_node.jpg)    
* size: The side length of image.
* border: The size of the border around the QR code, the larger the value, the wider the border.
* text: Enter the text content of the QR code here, and multi-language is not supported.

### <a id="table1">DecodeQRCode</a>
Decoding the QR code.

Node Options:  
![image](image/decode_qrcode_node.jpg)    
* image: The input QR code image.
* pre_blur: Pre-blurring, you can try to adjust this value for QR codes that are difficult to identify. 

### <a id="table1">LoadPSD</a>
![image](image/load_image_example_psd_file.jpg)    
![image](image/load_image_example.jpg)    
Load the PSD format file and export the layers.
Note that this node requires the installation of the ```psd_tools``` dependency package, If error occurs during the installation of psd_tool, such as ```ModuleNotFoundError: No module named 'docopt'``` , please download [docopt's whl](https://www.piwheels.org/project/docopt/) and manual install it. 


Node Options:  
![image](image/load_image_node.jpg)    
* image: Here is a list of *.psd files under ```ComfyUI/input```, where previously loaded psd images can be selected.
* file_path: The complete path and file name of the psd file.
* include_hidden_layer: whether include hidden layers.
* find_layer_by: The method for finding layers can be selected by layer key number or layer name. Layer groups are treated as one layer.
* layer_index: The layer key number, where 0 is the bottom layer, is incremented sequentially. If include_hiddenlayer is set to false, hidden layers are not counted. Set to -1 to output the top layer.
* layer_name: Layer name. Note that capitalization and punctuation must match exactly.

Outputs:
flat_image: PSD preview image.
layer_iamge: Find the layer output.
all_layers: Batch images containing all layers.


### <a id="table1">SD3NegativeConditioning</a>
![image](image/sd3_negative_conditioning_node_note.jpg)  
Encapsulate the four nodes of Negative Condition in SD3 into a separate node.

Node Options:  
![image](image/sd3_negative_conditioning_node.jpg)    
* zero_out_start: Set the ConditioningSetTimestepRange start value for Negative ConditioningZeroOut, which is the same as the ConditioningSetTimestepRange end value for Negative.


# <a id="table1">LayerMask</a>
![image](image/layermask_nodes.jpg)    


### <a id="table1">BlendIfMask</a>
Reproduction of Photoshop's layer Style - Blend If function. This node outputs a mask for layer composition on the ImageBlend or ImageBlendAdvance nodes.
```mask``` is an optional input, and if you enter a mask here, it will act on the output.
![image](image/blendif_mask_example.jpg)    

Node Options:  
![image](image/blendif_mask_node.jpg)    
* invert_mask: Whether to reverse the mask.
* blend_if: Channel selection for Blend If. There are four options: ```gray``` , ```red```, ```green```, and ```blue```.
* black_point: Black point values, ranging from 0-255.
* black_range: Dark part transition range. The larger the value, the richer the transition level of the dark part mask.
* white_point: White point values, ranging from 0-255.
* white_range: Brightness transition range. The larger the value is, the richer the transition level of the bright part mask is.


### <a id="table1">MaskBoxDetect</a>
Detect the area where the mask is located and output its position and size.
![image](image/mask_box_detect_example.jpg)    

Node options:  
![image](image/mask_box_detect_node.jpg)    
* detect: Detection method, ```min_bounding_rect``` is the minimum bounding rectangle of block shape, ```max_inscribed_rect``` is the maximum inscribed rectangle of block shape, and ```mask-area``` is the effective area for masking pixels.
* x_adjust: Adjust of horizontal deviation after detection.
* y_adjust: Adjust of vertical offset after detection.
* scale_adjust: Adjust the scaling offset after detection.

Output:
* box_preview: Preview image of detection results. Red represents the detected result, and green represents the adjust output result.
* x_percent: Horizontal position output in percentage.
* y_percent: Vertical position output in percentage.
* width: Width.
* height: Height.
* x: The x-coordinate of the top left corner position.
* y: The y-coordinate of the top left corner position.


## <a id="table1">Ultra</a>节点组
![image](image/ultra_nodes.jpg)   
Nodes that use ultra fine edge masking processing methods, the latest version of nodes includes: SegmentAnythingUltraV2, RmBgUltraV2, BiRefNetUltra, PersonMaskUltraV2, SegformerB2ClothesUltra and MaskEdgeUltraDetailV2.
There are three edge processing methods for these nodes:
* ```PyMatting``` optimizes the edges of the mask by using a closed form matching to mask trimap.
* ```GuideFilter``` uses opencv guidedfilter to feather edges based on color similarity, and performs best when edges have strong color separation.    
The code for the above two methods is from the [ComfyUI-Image-Filters](https://github.com/spacepxl/ComfyUI-Image-Filters) in spacepxl's Alpha Matte, thanks to the original author.
* ```VitMatte``` uses the transformer vit model for high-quality edge processing, preserving edge details and even generating semi transparent masks.
Note: When running for the first time, you need to download the vitmate model file and wait for the automatic download to complete. If the download cannot be completed, you can run the command ```huggingface-cli download hustvl/vitmatte-small-composition-1k``` to manually download.
After successfully downloading the model, you can use ```VITMatte(local)``` without accessing the network.
* VitMatte's options: ```device``` set whether to use CUDA for vitimate operations, which is about 5 times faster than CPU. ```max_megapixels``` set the maximum image size for vitmate operation, and oversized images will be reduced in size. For 16G VRAM, it is recommended to set it to 3.

Download the vitmatte model files [all files of vitmatte model](https://huggingface.co/hustvl/vitmatte-small-composition-1k/tree/main) to the ```ComfyUI/models/vitmatte``` folder.

The following figure is an example of the difference in output between three methods.
![image](image/mask_edge_ultra_detail_v2_example.jpg)   


### <a id="table1">SegmentAnythingUltra</a>
Improvements to [ComfyUI Segment Anything](https://github.com/storyicon/comfyui_segment_anything),  thanks to the original author.

*Please refer to the installation of ComfyUI Segment Anything to install the model. If ComfyUI Segment Anything has been correctly installed, you can skip this step.
* From [here](https://huggingface.co/bert-base-uncased/tree/main) download the config.json，model.safetensors，tokenizer_config.json，tokenizer.json and vocab.txt 5 files to ```ComfyUI/models/bert-base-uncased``` folder.
* Download [GroundingDINO_SwinT_OGC config file](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py), [GroundingDINO_SwinT_OGC model](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth), 
[GroundingDINO_SwinB config file](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py), [GroundingDINO_SwinB model](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth) to ```ComfyUI/models/grounding-dino``` folder.
* Download [sam_vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)，[sam_vit_l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth), 
[sam_vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth), [sam_hq_vit_h](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth),
[sam_hq_vit_l](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth), [sam_hq_vit_b](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth), 
[mobile_sam](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt) to ```ComfyUI/models/sams``` folder.

![image](image/segment_anything_ultra_compare.jpg)    
![image](image/segment_anything_ultra_example.jpg)    

Node options:  
![image](image/segment_anything_ultra_node.jpg)    
* sam_model: Select the SAM model.
* ground_dino_model: Select the Grounding DINO model.
* threshold: The threshold of SAM.
* detail_range: Edge detail range.
* black_point: Edge black sampling threshold.
* white_point: Edge white sampling threshold.
* process_detail: Set to false here will skip edge processing to save runtime.
* prompt: Input for SAM's prompt.

### <a id="table1">SegmentAnythingUltraV2</a>
The V2 upgraded version of SegmentAnythingUltra has added the VITMatte edge processing method.(Note: Images larger than 2K in size using this method will consume huge memory) 
![image](image/ultra_v2_nodes_example.jpg)    

On the basis of SegmentAnythingUltra, the following changes have been made: 
![image](image/segment_anything_ultra_v2_node.jpg)    
* detail_method: Edge processing methods. provides VITMatte, VITMatte(local), PyMatting, GuidedFilter. If the model has been downloaded after the first use of VITMatte, you can use VITMatte (local) afterwards.
* detail_erode: Mask the erosion range inward from the edge. the larger the value, the larger the range of inward repair.
* detail_dilate: The edge of the mask expands outward. the larger the value, the wider the range of outward repair.
* device: Set whether the VitMatte to use cuda.
* max_megapixels: Set the maximum size for VitMate operations.

### <a id="table1">Florence2Ultra</a>
Using the segmentation function of the Florence2 model, while also having ultra-high edge details.
The code for this node section is from [spacepxl/ComfyUI-Florence-2](https://github.com/spacepxl/ComfyUI-Florence-2), thanks to the original author.

![image](image/florence2_ultra_example.jpg)    

Node Options:  
![image](image/florence2_ultra_node.jpg)    
* florence2_model: Florence2 model input.
* image: Image input.
* task: Select the task for florence2.
* text_input: Text input for florence2.
* detail_method: Edge processing methods. provides VITMatte, VITMatte(local), PyMatting, GuidedFilter. If the model has been downloaded after the first use of VITMatte, you can use VITMatte (local) afterwards.
* detail_erode: Mask the erosion range inward from the edge. the larger the value, the larger the range of inward repair.
* detail_dilate: The edge of the mask expands outward. the larger the value, the wider the range of outward repair.
* black_point: Edge black sampling threshold.
* white_point: Edge white sampling threshold.
* process_detail: Set to false here will skip edge processing to save runtime.
* device: Set whether the VitMatte to use cuda.
* max_megapixels: Set the maximum size for VitMate operations.

### <a id="table1">LoadFlorence2Model</a>
Florence2 model loader.
*When using it for the first time, the model will be automatically downloaded.

![image](image/load_florence2_model_node.jpg)   
At present, there are base, base-ft, large, large-ft, DocVQA, SD3-Captioner and base-PromptGen models to choose from.


### <a id="table1">RemBgUltra</a>
Remove background. compared to the similar background removal nodes, this node has ultra-high edge details.

This node combines the Alpha Matte node of Spacepxl's [ComfyUI-Image-Filters](https://github.com/spacepxl/ComfyUI-Image-Filters) and the functionality of ZHO-ZHO-ZHO's [ComfyUI-BRIA_AI-RMBG](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG), thanks to the original author.

*Download the [BRIA Background Removal v1.4](https://huggingface.co/briaai/RMBG-1.4) model file (```model.pth```) to the ```/ComfyUI/models/rmbg/RMBG-1.4``` folder. This model was developed by BRIA AI and can be used as an open-source model for non-commercial purposes.    
**After downloading the model, recommended to add a line 'rmbg: models/rmbg/RMBG-1.4' in ComfyUI/extra_model_paths.yaml. (This step is not mandatory)    

![image](image/rembg_ultra_example.jpg)    

Node options:  
![image](image/rembg_ultra_node.jpg)    
* detail_range: Edge detail range.
* black_point: Edge black sampling threshold.
* white_point: Edge white sampling threshold.
* process_detail: Set to false here will skip edge processing to save runtime.

### <a id="table1">RmBgUltraV2</a>
The V2 upgraded version of RemBgUltra has added the VITMatte edge processing method.(Note: Images larger than 2K in size using this method will consume huge memory) 

On the basis of RemBgUltra, the following changes have been made: 
![image](image/rmbg_ultra_v2_node.jpg)    
* detail_method: Edge processing methods. provides VITMatte, VITMatte(local), PyMatting, GuidedFilter. If the model has been downloaded after the first use of VITMatte, you can use VITMatte (local) afterwards.
* detail_erode: Mask the erosion range inward from the edge. the larger the value, the larger the range of inward repair.
* detail_dilate: The edge of the mask expands outward. the larger the value, the wider the range of outward repair.
* device: Set whether the VitMatte to use cuda.
* max_megapixels: Set the maximum size for VitMate operations.

### <a id="table1">BiRefNetUltra</a>
Using the BiRefNet model to remove background has better recognition ability and ultra-high edge details.
The code for the model part of this node comes from Viper's [ComfyUI-BiRefNet](https://github.com/viperyl/ComfyUI-BiRefNet)，thanks to the original author.

*From [https://huggingface.co/ViperYX/BiRefNet](https://huggingface.co/ViperYX/BiRefNet/tree/main) download the ```BiRefNet-ep480.pth```,```pvt_v2_b2.pth```,```pvt_v2_b5.pth```,```swin_base_patch4_window12_384_22kto1k.pth```, ```swin_large_patch4_window12_384_22kto1k.pth``` 5 files to ```ComfyUI/models/BiRefNet``` folder.

![image](image/birefnet_ultra_example.jpg)    

Node options:  
![image](image/birefnet_ultra_node.jpg)    
* detail_method: Edge processing methods. provides VITMatte, VITMatte(local), PyMatting, GuidedFilter. If the model has been downloaded after the first use of VITMatte, you can use VITMatte (local) afterwards.
* detail_erode: Mask the erosion range inward from the edge. the larger the value, the larger the range of inward repair.
* detail_dilate: The edge of the mask expands outward. the larger the value, the wider the range of outward repair.
* black_point: Edge black sampling threshold.
* white_point: Edge white sampling threshold.
* process_detail: Set to false here will skip edge processing to save runtime.
* device: Set whether the VitMatte to use cuda.
* max_megapixels: Set the maximum size for VitMate operations.

### <a id="table1">TransparentBackgroundUltra</a>
Using the transparent-background model to remove background has better recognition ability and speed, while also having ultra-high edge details.

*From [googledrive](https://drive.google.com/drive/folders/10KBDY19egb8qEQBv34cqIVSwd38bUAa9?usp=sharing) or  [BaiduNetdisk](https://pan.baidu.com/s/1CYZ4VwvU06e-02BKRUj9QQ?pwd=jjt9)  download all files to ```ComfyUI/models/transparent-background``` folder.

![image](image/transparent_background_ultra_example.jpg)    

Node Options:  
![image](image/transparent_background_ultra_node.jpg)    
* model: Select the model.
* detail_method: Edge processing methods. provides VITMatte, VITMatte(local), PyMatting, GuidedFilter. If the model has been downloaded after the first use of VITMatte, you can use VITMatte (local) afterwards.
* detail_erode: Mask the erosion range inward from the edge. the larger the value, the larger the range of inward repair.
* detail_dilate: The edge of the mask expands outward. the larger the value, the wider the range of outward repair.
* black_point: Edge black sampling threshold.
* white_point: Edge white sampling threshold.
* process_detail: Set to false here will skip edge processing to save runtime.
* device: Set whether the VitMatte to use cuda.
* max_megapixels: Set the maximum size for VitMate operations.

### <a id="table1">PersonMaskUltra</a>
Generate masks for portrait's face, hair, body skin, clothing, or accessories. Compared to the previous A Person Mask Generator node, this node has ultra-high edge details.
The model code for this node comes from [a-person-mask-generator](https://github.com/djbielejeski/a-person-mask-generator)， edge processing code from [ComfyUI-Image-Filters](https://github.com/spacepxl/ComfyUI-Image-Filters)，thanks to the original author.

![image](image/person_mask_ultra_example.jpg)    

Node options:  
![image](image/person_mask_ultra_node.jpg)    
* face: Face recognition.
* hair: Hair recognition.
* body: Body skin recognition.
* clothes: Clothing recognition.
* accessories: Identification of accessories (such as backpacks).
* background: Background recognition.
* confidence: Recognition threshold, lower values will output more mask ranges.
* detail_range: Edge detail range.
* black_point: Edge black sampling threshold.
* white_point: Edge white sampling threshold.
* process_detail: Set to false here will skip edge processing to save runtime.

### <a id="table1">PersonMaskUltraV2</a>
The V2 upgraded version of PersonMaskUltra has added the VITMatte edge processing method.(Note: Images larger than 2K in size using this method will consume huge memory) 

On the basis of PersonMaskUltra, the following changes have been made: 
![image](image/person_mask_ultra_v2_node.jpg)    
* detail_method: Edge processing methods. provides VITMatte, VITMatte(local), PyMatting, GuidedFilter. If the model has been downloaded after the first use of VITMatte, you can use VITMatte (local) afterwards.
* detail_erode: Mask the erosion range inward from the edge. the larger the value, the larger the range of inward repair.
* detail_dilate: The edge of the mask expands outward. the larger the value, the wider the range of outward repair.
* device: Set whether the VitMatte to use cuda.
* max_megapixels: Set the maximum size for VitMate operations.
* 
### <a id="table1">SegformerB2ClothesUltra</a>
![image](image/segformer_ultra_example.jpg)   
Generate masks for characters' faces, hair, arms, legs, and clothing, mainly used for segmenting clothing.
The model segmentation code is from[StartHua](https://github.com/StartHua/Comfyui_segformer_b2_clothes)，thanks to the original author.
Compared to the comfyui_segformer_b2_clothes, this node has ultra-high edge details. (Note: Generating images with edges exceeding 2K in size using the VITMatte method will consume a lot of memory)    

*Download all model files from [here](https://huggingface.co/mattmdjaga/segformer_b2_clothes/tree/main) to   ```ComfyUI/models/segformer_b2_clothes``` folder.

Node Options:  
![image](image/segformer_ultra_node.jpg)    
* face: Facial recognition switch.
* hair: Hair recognition switch.
* hat: Hat recognition switch.
* sunglass: Sunglass recognition switch.
* left_arm: Left arm recognition switch.
* right_arm: Right arm recognition switch.
* left_leg: Left leg recognition switch.
* right_leg: Right leg recognition switch.
* skirt: Skirt recognition switch.
* pants: Pants recognition switch.
* dress: Dress recognition switch.
* belt: Belt recognition switch.
* shoe: Shoes recognition switch.
* bag: Bag recognition switch.
* scarf: Scarf recognition switch.
* detail_method: Edge processing methods. provides VITMatte, VITMatte(local), PyMatting, GuidedFilter. If the model has been downloaded after the first use of VITMatte, you can use VITMatte (local) afterwards.
* detail_erode: Mask the erosion range inward from the edge. the larger the value, the larger the range of inward repair.
* detail_dilate: The edge of the mask expands outward. the larger the value, the wider the range of outward repair.
* black_point: Edge black sampling threshold.
* white_point: Edge white sampling threshold.
* process_detail: Set to false here will skip edge processing to save runtime.
* device: Set whether the VitMatte to use cuda.
* max_megapixels: Set the maximum size for VitMate operations.

### <a id="table1">SegformerUltraV2</a>
![image](image/segformer_clothes_example.jpg)   
![image](image/segformer_fashion_example.jpg)   
Using the segformer model to segment clothing with ultra-high edge details. Currently supports segformer b2 clothes, segformer b3 clothes and segformer b3 fashion。

*from [here](https://huggingface.co/mattmdjaga/segformer_b2_clothes/tree/main) download all files to ```ComfyUI/models/segformer_b2_clothes``` folder.        
*from [here](https://huggingface.co/sayeed99/segformer_b3_clothes/tree/main) download all files to ```ComfyUI/models/segformer_b3_clothes``` folder.    
*from [here](https://huggingface.co/sayeed99/segformer-b3-fashion/tree/main) download all files to ```ComfyUI/models/segformer_b3_fashion``` folder.    

Node Options:  
![image](image/segformer_ultra_v2_node.jpg)    
* image: The input image.
* segformer_pipeline: Segformer pipeline input. The pipeline is output by SegformerClottesPipeline and SegformerFashionPipeline node.
* detail_method: Edge processing methods. provides VITMatte, VITMatte(local), PyMatting, GuidedFilter. If the model has been downloaded after the first use of VITMatte, you can use VITMatte (local) afterwards.
* detail_erode: Mask the erosion range inward from the edge. the larger the value, the larger the range of inward repair.
* detail_dilate: The edge of the mask expands outward. the larger the value, the wider the range of outward repair.
* black_point: Edge black sampling threshold.
* white_point: Edge white sampling threshold.
* process_detail: Set to false here will skip edge processing to save runtime.
* device: Set whether the VitMatte to use cuda.
* max_megapixels: Set the maximum size for VitMate operations.

### <a id="table1">SegformerClothesPipiline</a>
Select the segformer clothes model and choose the segmentation content.  

Node Options:  
![image](image/segformer_clothes_pipeline_node.jpg)    
* model: Model selection. There are currently two models available to choose from for segformer b2 clothes and segformer b3 clothes.
* face: Facial recognition switch.
* hair: Hair recognition switch.
* hat: Hat recognition switch.
* sunglass: Sunglass recognition switch.
* left_arm: Left arm recognition switch.
* right_arm: Right arm recognition switch.
* left_leg: Left leg recognition switch.
* right_leg: Right leg recognition switch.
* left_shoe: Left shoe recognition switch.
* right_shoe: Right shoe recognition switch.
* skirt: Skirt recognition switch.
* pants: Pants recognition switch.
* dress: Dress recognition switch.
* belt: Belt recognition switch.
* bag: Bag recognition switch.
* scarf: Scarf recognition switch.

### <a id="table1">SegformerFashionPipiline</a> 
Select the segformer fashion model and choose the segmentation content.

Node Options:  
![image](image/segformer_fashion_pipeline_node.jpg)    
* model: Model selection. Currently, there is only one model available for selection: segformer b3 fashion。
* shirt: shirt and blouse switch.
* top: top, t-shirt, sweatshirt switch.
* sweater: sweater switch.
* cardigan: cardigan switch.
* jacket: jacket switch.
* vest: vest switch.
* pants: pants switch.
* shorts: shorts switch.
* skirt: skirt switch.
* coat: coat switch.
* dress: dress switch.
* jumpsuit: jumpsuit switch.
* cape: cape switch.
* glasses: glasses switch.
* hat: hat switch.
* hairaccessory: headband, head covering, hair accessory switch.
* tie: tie switch.
* glove: glove switch.
* watch: watch switch.
* belt: belt switch.
* legwarmer: leg warmer switch.
* tights: tights and stockings switch.
* sock: sock switch.
* shoe: shoes switch.
* bagwallet: bag and wallet switch.
* scarf: scarf switch.
* umbrella: umbrella switch.
* hood: hood switch.
* collar: collar switch.
* lapel: lapel switch.
* epaulette: epaulette switch.
* sleeve: sleeve switch.
* pocket: pocket switch.
* neckline: neckline switch.
* buckle: buckle switch.
* zipper: zipper switch.
* applique: applique switch.
* bead: bead switch.
* bow: bow switch.
* flower: flower switch.
* fringe: fringe switch.
* ribbon: ribbon switch.
* rivet: rivet switch.
* ruffle: ruffle switch.
* sequin: sequin switch.
* tassel: tassel switch.

### <a id="table1">MaskEdgeUltraDetail</a>
Process rough masks to ultra fine edges.
This node combines the Alpha Matte and the Guided Filter Alpha nodes functions of Spacepxl's [ComfyUI-Image-Filters](https://github.com/spacepxl/ComfyUI-Image-Filters), thanks to the original author.
![image](image/mask_edge_ultra_detail_example.jpg)    

Node options:  
![image](image/mask_edge_ultra_detail_node.jpg)    
* method: Provide two methods for edge processing: PyMatting and OpenCV-GuidedFilter. PyMatching has a slower processing speed, but for video, it is recommended to use this method to obtain smoother mask sequences.
* mask_grow: Mask expansion amplitude. positive values expand outward, while negative values contract inward. For rougher masks, negative values are usually used to shrink their edges for better results.
* fix_gap: Repair the gaps in the mask. if obvious gaps in the mask, increase this value appropriately.
* fix_threshold: The threshold of fix_gap.
* detail_range: Edge detail range.
* black_point: Edge black sampling threshold.
* white_point: Edge white sampling threshold.

### <a id="table1">MaskEdgeUltraDetailV2</a>
The V2 upgraded version of MaskEdgeUltraDetail has added the VITMatte edge processing method.(Note: Images larger than 2K in size using this method will consume huge memory)    
This method is suitable for handling semi transparent areas. 

On the basis of MaskEdgeUltraDetail, the following changes have been made: 
![image](image/mask_edge_ultra_detail_v2_node.jpg)    
* method: Edge processing methods. provides VITMatte, VITMatte(local), PyMatting, GuidedFilter. If the model has been downloaded after the first use of VITMatte, you can use VITMatte (local) afterwards.
* edge_erode: Mask the erosion range inward from the edge. the larger the value, the larger the range of inward repair.
* edge_dilate: The edge of the mask expands outward. the larger the value, the wider the range of outward repair.
* device: Set whether the VitMatte to use cuda.
* max_megapixels: Set the maximum size for VitMate operations.

### <a id="table1">YoloV8Detect</a>
Use the YoloV8 model to detect faces, hand box areas, or character segmentation. Supports the output of the selected number of channels.
Download the model files from [GoogleDrive](https://drive.google.com/drive/folders/1I5TISO2G1ArSkKJu1O9b4Uvj3DVgn5d2) or [BaiduNetdisk](https://pan.baidu.com/s/1ImoJrzL1zDgaCqaSzrNEtw?pwd=5xgk) to ```ComfyUI/models/yolo``` folder.

![image](image/yolov8_detect_example.jpg)    

Node Options:  
![image](image/yolov8_detect_node.jpg)    
* yolo_model: Yolo model selection. the model with ```seg``` name can output segmented masks, otherwise they can only output box masks.
* mask_merge: Select the merged mask. ```all``` is to merge all mask outputs. The selected number is how many masks to output, sorted by recognition confidence to merge the output.

Outputs:
* mask: The output mask.
* yolo_plot_image: Preview of yolo recognition results.
* yolo_masks: For all masks identified by yolo, each individual mask is output as a mask.


### <a id="table1">MediapipeFacialSegment</a>
Use the Mediapipe model to detect facial features, segment left and right eyebrows, eyes, lips, and tooth.
![image](image/mediapipe_facial_segment_example.jpg)    

Node Options:  
![image](image/mediapipe_facial_segment_node.jpg)    
* left_eye: Recognition switch of left eye.
* left_eyebrow: Recognition switch of left eyebrow.
* right_eye: Recognition switch of right eye.
* right_eyebrow: Recognition switch of right eyebrow.
* lips: Recognition switch of lips.
* tooth: Recognition switch of tooth.

### <a id="table1">MaskByColor</a>
Generate a mask based on the selected color.
![image](image/mask_by_color_example.jpg)    

Node Options:  
![image](image/mask_by_color_node.jpg)    
* image: Input image.
* mask: This input is optional, if there is a mask, only the colors inside the mask are included in the range.
* color: Color selector. Click on the color block to select a color, and you can use the straws on the color picker panel to pick up the screen color. Note: When using straws, maximize the browser window.
* color_in_HEX<sup>4</sup>: Enter color values. If this item has input, it will be used first, ignoring the color selected by the ```color``` .
* threshold: Mask range threshold, the larger the value, the larger the mask range.
* fix_gap: Repair the gaps in the mask. If there are obvious gaps in the mask, increase this value appropriately.
* fix_threshold: The threshold for repairing masks. 
* invert_mask: Whether to reverse the mask.

### <a id="table1">ImageToMask</a>
Convert the image to a mask. Supports converting any channel in LAB, RGBA, YUV, and HSV modes into masks, while providing color scale adjustment. Support mask optional input to obtain masks that only include valid parts.
![image](image/image_to_mask_example.jpg)    

Node Options:  
![image](image/image_to_mask_node.jpg)    
* image: Input image.
* mask: This input is optional, if there is a mask, only the colors inside the mask are included in the range.
* channel: Channel selection. You can choose any channel of LAB, RGBA, YUV, or HSV modes.
* black_point<sup>*</sup>: Black dot value for the mask. The value range is 0-255, with a default value of 0.
* white_point<sup>*</sup>: White dot value for the mask. The value range is 0-255, with a default value of 255.
* gray_point: Gray dot values for the mask. The value range is 0.01-9.99, with a default of 1.
* invert_output_mask: Whether to reverse the mask.

<sup>*</sup><font size="3">If the black_point or output_black_point value is greater than white_point or output_white_point, the two values are swapped, with the larger value used as white_point and the smaller value used as black_point.</font>      


### <a id="table1">Shadow</a> & Highlight Mask
Generate masks for the dark and bright parts of the image.
![image](image/shadow_and_highlight_mask_example.jpg)    

Node options:  
![image](image/shadow_and_highlight_mask_node.jpg)    
* image: The input image.
* mask: Optional input. if there is input, only the colors within the mask range will be adjusted.
* shadow_level_offset: The offset of values in the dark area, where larger values bring more areas closer to the bright into the dark area.
* shadow_range: The transitional range of the dark area.
* highlight_level_offset: The offset of values in the highlight area, where larger values bring more areas closer to the dark into the highlight area.
* highlight_range: The transitional range of the highlight area. 


### <a id="table1">PixelSpread</a>
Pixel expansion preprocessing on the masked edge of an image can effectively improve the edges of image composit.
![image](image/pixel_spread_example.jpg)    

Node options:  
![image](image/pixel_spread_node.jpg)    
* invert_mask: Whether to reverse the mask.
* mask_grow: Mask expansion amplitude.

### <a id="table1">MaskByDifferent</a>
Calculate the differences between two images and output them as mask.
![image](image/mask_by_different_example.jpg)    

Node options:  
![image](image/mask_by_different_node.jpg)    
* gain: The gain of difference calculate. higher value will result in a more significant slight difference.
* fix_gap: Fix the internal gaps of the mask. higher value will repair larger gaps.
* fix_threshold: The threshold for fix_gap.
* main_subject_detect: Setting this to True will enable subject detection, ignoring differences outside of the subject.

### <a id="table1">MaskGrow</a>
Grow and shrink edges and blur the mask
![image](image/mask_grow_example.jpg)    

Node options:
![image](image/mask_grow_node.jpg)    
* invert_mask: Whether to reverse the mask.
* grow: Positive values expand outward, while negative values contract inward.
* blur: Blur the edge.

### <a id="table1">MaskEdgeShrink</a>
Smooth transition and shrink the mask edges while preserving edge details.
![image](image/mask_edge_shrink_example.jpg)    

Node options:
![image](image/mask_edge_shrink_node.jpg)    
* invert_mask: Whether to reverse the mask.
* shrink_level: Shrink the smoothness level.
* soft: Smooth amplitude.
* edge_shrink: Edge shrinkage amplitude.
* edge_reserve: Preserve the amplitude of edge details, 100 represents complete preservation, and 0 represents no preservation at all.

Comparison of MaskGrow and MaskEdgeShrink
![image](image/mask_edge_compare.jpg)    

### <a id="table1">MaskMotionBlur</a>
Create motion blur on the mask.
![image](image/mask_motion_blur_example.jpg)    

Node options:  
![image](image/mask_motion_blur_node.jpg)    
* invert_mask: Whether to reverse the mask.
* blur: The size of blur.
* angle: The angle of blur.

### <a id="table1">MaskGradient</a>
Create a gradient for the mask from one side. please note the difference between this node and the CreateGradientMask node.
![image](image/mask_gradient_example.jpg)    

Node options:
![image](image/mask_gradient_node.jpg)    
* invert_mask: Whether to reverse the mask.
* gradient_side: Generate gradient from which edge. There are four directions: top, bottom, left and right.
* gradient_scale: Gradient distance. The default value of 100 indicates that one side of the gradient is completely transparent and the other side is completely opaque. The smaller the value, the shorter the distance from transparent to opaque.
* gradient_offset: Gradient position offset.
* opacity: The opacity of the gradient.

### <a id="table1">CreateGradientMask</a>
Create a gradient mask. please note the difference between this node and the MaskGradient node.
![image](image/create_gradient_mask_example.jpg)    
![image](image/create_gradient_mask_example2.jpg)    

Node options:  
![image](image/create_gradient_mask_node.jpg)    
* size_as<sup>*</sup>: The input image or mask here will generate the output image and mask according to their size. this input takes priority over the width and height below.
* width: Width of the image. If there is a size_as input, this setting will be ignored.
* height: Height of the image. If there is a size_as input, this setting will be ignored.
* gradient_side: Generate gradient from which edge. There are five directions: top, bottom, left, right and center.
* gradient_scale: Gradient distance. The default value of 100 indicates that one side of the gradient is completely transparent and the other side is completely opaque. The smaller the value, the shorter the distance from transparent to opaque.
* gradient_offset: Gradient position offset. When ```gradient_side``` is center, the size of the gradient area is adjusted here, positive values are smaller, and negative values are enlarged.
* opacity: The opacity of the gradient.

<sup>*</sup>Only limited to input image and mask. forcing the integration of other types of inputs will result in node errors.  


### <a id="table1">MaskStroke</a>
Generate mask contour strokes.
![image](image/mask_stroke_example.jpg)    

Node options:  
![image](image/mask_stroke_node.jpg)    
* invert_mask: Whether to reverse the mask.
* stroke_grow: Stroke expansion/contraction amplitude, positive values indicate expansion and negative values indicate contraction.
* stroke_width: Stroke width.
* blur: Blur of stroke.

### <a id="table1">MaskGrain</a>
Generates noise for the mask.
![image](image/mask_grain_example.jpg)    

Node Options:  
![image](image/mask_grain_node.jpg)    
* grain: Noise intensity.
* invert_mask: Whether to reverse the mask.



### <a id="table1">MaskPreview</a>
Preview the input mask
![image](image/mask_invert.jpg)    


### <a id="table1">MaskInvert</a>
Invert the mask
![image](image/mask_invert_node.jpg)    


# <a id="table1">LayerFilter</a>
![image](image/layerfilter_nodes.jpg)    

### <a id="table1">Sharp</a> & Soft
Enhance or smooth out details for image.
![image](image/sharp_and_soft_example.jpg)    

Node options:
![image](image/sharp_and_soft_node.jpg)    
* enhance: Provide 4 presets, which are very sharp, sharp, soft and very soft. If you choose None, you will not do any processing.

### <a id="table1">SkinBeauty</a>
Make the skin look smoother.
![image](image/skin_beauty_example.jpg)    

Node options:
![image](image/skin_beauty_node.jpg)    
* smooth: Skin smoothness.
* threshold: Smooth range. the larger the range with the smaller value.
* opacity: The opacity of the smoothness.


### <a id="table1">WaterColor</a>
Watercolor painting effect
![image](image/water_color_example.jpg)    

Node option:
![image](image/water_color_node.jpg)    
* line_density: The black line density.
* opacity: The opacity of watercolor effects.


### <a id="table1">SoftLight</a>
Soft light effect, the bright highlights on the screen appear blurry.
![image](image/soft_light_example.jpg)    

Node options:
![image](image/soft_light_node.jpg)    
* soft: Size of soft light.
* threshold: Soft light range. the light appears from the brightest part of the picture. in lower value, the range will be larger, and in higher value, the range will be smaller.
* opacity: Opacity of the soft light.


### <a id="table1">ChannelShake</a>
Channel misalignment. similar to the effect of Tiktok logo.
![image](image/channel_shake_example.jpg)    

Node options:
![image](image/channel_shake_node.jpg)    
* distance: Distance of channel separation.
* angle: Angle of channel separation.
* mode: Channel shift arrangement order.


### <a id="table1">HDR</a> Effects
enhances the dynamic range and visual appeal of input images.
This node is reorganize and encapsulate of  [HDR Effects (SuperBeasts.AI)](https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts), thanks to the original author.
![image](image/hdr_effects_example.jpg)    

Node options:
![image](image/hdr_effects_node.jpg)    
* hdr_intensity: Range: 0.0 to 5.0, Controls the overall intensity of the HDR effect, Higher values result in a more pronounced HDR effect.
* shadow_intensity: Range: 0.0 to 1.0，Adjusts the intensity of shadows in the image，Higher values darken the shadows and increase contrast.
* highlight_intensity: Range: 0.0 to 1.0，Adjusts the intensity of highlights in the image，Higher values brighten the highlights and increase contrast.
* gamma_intensity: Range: 0.0 to 1.0，Controls the gamma correction applied to the image，Higher values increase the overall brightness and contrast.
* contrast: Range: 0.0 to 1.0，Enhances the contrast of the image, Higher values result in more pronounced contrast.
* enhance_color: Range: 0.0 to 1.0，Enhances the color saturation of the image, Higher values result in more vibrant colors.


### <a id="table1">Film</a>
Simulate the grain, dark edge, and blurred edge of the film, support input depth map to simulate defocus.    
This node is reorganize and encapsulate of [digitaljohn/comfyui-propost](https://github.com/digitaljohn/comfyui-propost), thanks to the original author.
![image](image/film_example.jpg)    

Node options:
![image](image/film_node.jpg)    
* image: The input image.
* depth_map: Input depth map to simulate defocus effect. it is an optional input. if there is no input, will simulates radial blur at the edges of the image.
* center_x: The horizontal axis of the center point position of the dark edge and radial blur, where 0 represents the leftmost side, 1 represents the rightmost side, and 0.5 represents at the center.
* center_y: The vertical axis of the center point position of the dark edge and radial blur, where 0 represents the leftmost side, 1 represents the rightmost side, and 0.5 represents at the center.
* saturation: Color saturation, 1 is the original value.
* grain_power: Grain intensity. larger value means more pronounced the noise.
* grain_scale: Grain size.
* grain_sat: The color saturation of grain. 0 represents mono noise, and the larger the value, the more prominent the color.
* grain_shadows: Grain intensity of dark part.
* grain_highs: Grain intensity of light part.
* blur_strength: The strength of blur. larger value means more blurry it becomes.
* blur_focus_spread: Focus diffusion range. larger value means larger clear range.
* focal_depth: Simulate the focal distance of defucus. 0 indicates that focus is farthest, and 1 indicates that is closest. this setting only valid when input the depth_map.

### <a id="table1">FilmV2</a>
The upgraded version of the Film node adds the fastgrain method on the basis of the previous one, and the speed of generating noise is accelerated by 10 times. The code for fastgrain is from [github.com/spacepxl/ComfyUI-Image-Filters](https://github.com/spacepxl/ComfyUI-Image-Filters) BetterFilmGrain node, thanks to the original authors.
![image](image/film_v2_node.jpg)    

### <a id="table1">LightLeak</a>
Simulate the light leakage effect of the film. please download the [light_leak.pkl(Baidu Netdisk)](https://pan.baidu.com/s/1QY1ZyYm885krrqDB2t8lPg?pwd=yvs3) or [light_leak.pkl(Google Drive)]([light_leak.pkl(Google Drive)(https://drive.google.com/file/d/1DcH2Zkyj7W3OiAeeGpJk1eaZpdJwdCL-/view?usp=sharing)), copy the file to ```ComfyUI/models/layerstyle``` folder.
![image](image/light_leak_example.jpg)    

Node options:
![image](image/light_leak_node.jpg)    
* light: 32 types of light spots are provided. random is a random selection.
* corner: There are four options for the corner where the light appears: top left, top right, bottom left, and bottom right.
* hue: The hue of the light.
* saturation: The color saturation of the light.
* opacity: The opacity of the light.

### <a id="table1">ColorMap</a>
Pseudo color heat map effect.
![image](image/colormap_result.jpg)    

Node options:
![image](image/color_map_node.jpg)    
* color_map: Effect type. there are a total of 22 types of effects, as shown in the above figure.
* opacity: The opacity of the color map effect.


### <a id="table1">MotionBlur</a>
Make the image motion blur
![image](image/motion_blur_example.jpg)    

Node options:
* angle: The angle of blur.
* blur: The size of blur.

### <a id="table1">GaussianBlur</a>
Make the image gaussian blur
![image](image/gaussian_blur_example.jpg)    

Node options:
* blur: The size of blur.

### <a id="table1">AddGrain</a>
Add noise to the picture.
![image](image/add_grain_example.jpg)    

Node Options:
![image](image/add_grain_node.jpg)    
* grain_power: Noise intensity.
* grain_scale: Noise size.
* grain_sat: Color saturation of noise.


## Annotation for <a id="table1">notes</a>
<sup>1</sup>  The layer_image, layer_mask and the background_image(if have input), These three items must be of the same size.    

<sup>2</sup>  The mask not a mandatory input item. the alpha channel of the image is used by default. If the image input does not include an alpha channel, the entire image's alpha channel will be automatically created. if have masks input simultaneously, the alpha channel will be overwrite by the mask.    

<sup>3</sup>  The <a id="table1">Blend</a> Mode include **normal, multply, screen, add, subtract, difference, darker, color_burn, color_dodge, linear_burn, linear_dodge, overlay, soft_light, hard_light, vivid_light, pin_light, linear_light, and hard_mix.** all of 19 blend modes in total.    
![image](image/blend_mode_result.jpg)    
<font size="1">*Preview of the blend mode  </font><br />     

<sup>3</sup>   The <a id="table1">BlendModeV2</a> include **normal, dissolve, darken, multiply, color burn, linear burn, darker color, lighten, screen, color dodge, linear dodge(add), lighter color, dodge, overlay, soft light, hard light, vivid light, linear light, pin light, hard mix, difference, exclusion, subtract, divide, hue, saturation, color, luminosity, grain extract, grain merge** all of 30 blend modes in total.      
Part of the code for BlendMode V2 is from [Virtuoso Nodes for ComfyUI](https://github.com/chrisfreilich/virtuoso-nodes). Thanks to the original authors.
![image](image/blend_mode_v2_example.jpg)    
<font size="1">*Preview of the Blend Mode V2</font><br />     

<sup>4</sup>  The RGB color described by hexadecimal RGB format, like '#FA3D86'.    

<sup>5</sup>  The layer_image and layer_mask must be of the same size.    

## Stars 

[![Star History Chart](https://api.star-history.com/svg?repos=chflame163/ComfyUI_LayerStyle&type=Date)](https://star-history.com/#chflame163/ComfyUI_LayerStyle&Date)

# statement
LayerStyle nodes follows the MIT license, Some of its functional code comes from other open-source projects. Thanks to the original author. If used for commercial purposes, please refer to the original project license to authorization agreement.

