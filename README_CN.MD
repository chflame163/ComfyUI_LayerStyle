# ComfyUI Layer Style
一组为ComfyUI设计的节点，可以合成图层达到类似Photoshop的功能。这些节点将PhotoShop的一部分基本功能迁移到ComfyUI，旨在集中工作流程，减少软件切换的频率。

![image](image/title.png)
节点按照功能分为5组：LayerStyle, LayerColor, LayerMask, LayerUtility和LayerFilter。
* [LayerStyle](#LayerStyle)节点组提供仿照Adobe Photoshop的图层样式。
* [LayerColor](#LayerColor)节点组提供调整颜色功能。
* [LayerMask](#LayerMask)节点组提供Mask辅助工具。
* [LayerUtility](#LayerUtility)节点组提供图层合成工具和工作流相关的辅助节点。
* [LayerFilter](#LayerFilter)节点组提供图像效果滤镜。

## 更新说明
**如果本插件更新后出现依赖包错误，请重新安装相关依赖包。详情见[这里](https://github.com/chflame163/ComfyUI_LayerStyle/issues/5)。
* 添加[MaskEdgeUltraDetail](#MaskEdgeUltraDetail) 节点，给粗糙的遮罩进行处理获得超精细的边缘。添加 [Exposure](#Exposure) 节点，调整图像曝光。
* 添加[Sharp & Soft](#Sharp) 节点，可提升或抹平图像细节。新增[MaskByDifferent](#MaskByDifferent)节点，比较两张图片并输出Mask。新增[SegmentAnythingUltra](#SegmentAnythingUltra)节点，提升遮罩边缘质量。*如果没有安装SegmentAnything, 需要手动下载模型。
* 所有节点已全面支持批量图片，为创作视频提供方便。( CropByMask 节点仅支持相同尺寸的切除, 如果输入批量mask_for_crop，将使用第一张的数据。)
* 添加[RemBgUltra](#RemBgUltra) 和 [PixelSpread](#PixelSpread) 节点，显著提升了遮罩质量。*RemBgUltra需手动下载模型。
* 添加[TextImage](#TextImage) 节点，生成文字图像和遮罩。
* 图像之间的[混合模式](#混合模式)增加新类型，现在支持多达19种混合模式。新增color_burn颜色加深, color_dodge颜色减淡, linear_burn线性加深, linear_dodge线性减淡, overlay叠加, soft_light柔光, hard_light强光, vivid_light亮光, pin_light点光, linear_light线性光, hard_mix实色混合。新增的混合模式适用于所有支持混合模式的节点。
* 添加[ColorMap](#ColorMap) 滤镜节点，用于制作伪彩色热力图效果。
* 添加[WaterColor](#WaterColor) 和 [SkinBeauty](#SkinBeauty) 节点。这是两个图像滤镜，生成水彩画和磨皮效果。
* 添加[ImageShift](#ImageShift) 节点，使图片产生位移，可输出位移接缝遮罩，方便制作连续贴图。
* 添加[ImageMaskScaleAs](#ImageMaskScaleAs) 节点，可根据参考图片调整图像或遮罩大小。
* 添加[ImageScaleRestore](#ImageScaleRestore) 节点，用于配合CropByMask进行局部放大修复工作。
* 添加[CropByMask](#CropByMask) 和 [RestoreCropBox](#RestoreCropBox)节点。此二者配合可将图片局部裁切重绘然后还原。
* 添加[ColorAdapter](#ColorAdapter) 节点，可自动调整图片色调。
* 添加[MaskStroke](#MaskStroke) 节点，可产生mask描边。
* 添加[LayerColor](#LayerColor)节点组，用于调整图像颜色。包括[LUT Apply](#LUT)，[Gamma](#Gamma), [Brightness & Contrast](#Brightness), [RGB](#RGB), [YUV](#YUV), [LAB](#LAB)和[HSV](#HSV)。
* 添加[ImageChannelSplit](#ImageChannelSplit)和[ImageChannelMerge](#ImageChannelMerge)节点。
* 添加[MaskMotionBlur](#MaskMotionBlur)节点。
* 添加[SoftLight](#SoftLight)节点。
* 添加[ChannelShake](#ChannelShake)节点，这是一个滤镜，能产生类似抖音logo的通道错位效果。
* 添加[MaskGradient](#MaskGradient)节点，可使mask产生渐变。
* 添加[GetColorTone](#GetColorTone)节点，可以获取图片的主色或平均色。添加[MaskGrow](#MaskGrow)和[MaskEdgeShrink](#MaskEdgeShrink)节点。

* 添加[MaskBoxDetect](#MaskBoxDetect)节点，可以通过mask自动探测位置并输出到合成节点。添加[XY to Percent](#Percent)节点，将绝对坐标转换为percent坐标。添加[GaussianBlur](#GaussianBlur)节点。添加[GetImageSize](#GetImageSize)节点。
* 添加[ExtendCanvas](#ExtendCanvas)节点。
* 添加[ImageBlendAdvance](#ImageBlendAdvance)节点。这个节点允许合成尺寸不同的背景图和图层，提供了更加自由的合成体验。
  添加[PrintInfo](#PrintInfo)节点作为工作流调试辅助工具。
* 添加[ColorImage](#ColorImage)和[GradientImage](#GradientImage)节点，用于生成纯色和渐变色图像。
* 添加[GradientOverlay](#GradientOverlay),[ColorOverlay](#ColorOverlay)节点。增加无效mask输入判断，当输入无效mask时将其忽略。
* 添加[InnerGlow](#InnerGlow), [InnerShadow](#InnerShadow), [MotionBlur](#MotionBlur)节点。
* 所有已完成的节点重新命名，节点分为4组：LayerStyle, LayerMask, LayerUtility, LayerFilter。
  因为重新命名，包含旧版节点的工作流需手动替换新版节点。
* [OuterGlow](#OuterGlow)节点修改，增加亮度、灯光颜色、辉光颜色选项。
* 添加[MaskInvert](#MaskInvert)节点。
* 添加[Stroke](#Stroke)节点。
* 添加[MaskPreview](#MaskPreview)节点。
* 添加[ImageOpacity](#ImageOpacity)节点。
* layer_mask修改为非必选, 默认使用layer_image的alpha通道，允许通过输入mask改变之，但是尺寸必须一致。
* 添加[ImageBlend](#ImageBlend)节点。
* 添加[OuterGlow](#OuterGlow)节点。
* 首个节点[DropShadow](#DropShadow)提交。


# <a id="table1">LayerStyle</a>
![image](image/layerstyle_title.png)
![image](image/layerstyle_nodes.png)


### <a id="table1">DropShadow</a>
生成阴影。
![image](image/drop_shadow_example.png)

节点选项说明:   
![image](image/drop_shadow_node.png)
* background_image<sup>1</sup>: 背景图像。
* layer_image<sup>1</sup>: 用于合成的层图像。
* layer_mask<sup>1,2</sup>: 层图像的遮罩，阴影按此生成。
* invert_mask: 是否反转遮罩。
* blend_mode<sup>3</sup>: 阴影的混合模式。
* opacity: 阴影的不透明度。
* distance_x: 阴影的水平方向偏移量。
* distance_y: 阴影的垂直方向偏移量。
* grow: 阴影扩张幅度。
* blur:阴影模糊程度。
* shadow_color<sup>4</sup>: 阴影颜色。
* [节点注解](#节点注解)


### <a id="table1">OuterGlow</a>
生成外发光。
![image](image/outer_glow_example.png)

节点选项说明:   
![image](image/outer_glow_node.png)
* background_image<sup>1</sup>: 背景图像。
* layer_image<sup>1</sup>: 用于合成的层图像。
* layer_mask<sup>1,2</sup>: 层图像的遮罩，外发光按此生成。
* invert_mask: 是否反转遮罩。
* blend_mode<sup>3</sup>: 发光的混合模式。
* opacity: 发光的不透明度。
* brightness: 发光亮度。
* glow_range: 发光范围。
* blur:发光模糊程度。
* light_color<sup>4</sup>: 发光中心颜色。
* glow_colo<sup>4</sup>: 辉光外围颜色。
* [节点注解](#节点注解)


### <a id="table1">InnerShadow</a>
生成内阴影。
![image](image/inner_shadow_example.png)

节点选项说明:   
![image](image/inner_shadow_node.png)
* background_image<sup>1</sup>: 背景图像。
* layer_image<sup>1</sup>: 用于合成的层图像。
* layer_mask<sup>1,2</sup>: 层图像的遮罩，阴影按此生成。
* invert_mask: 是否反转遮罩。
* blend_mode<sup>3</sup>: 阴影的混合模式。
* opacity: 阴影的不透明度。
* distance_x: 阴影的水平方向偏移量。
* distance_y: 阴影的垂直方向偏移量。
* grow: 阴影扩张幅度。
* blur:阴影模糊程度。
* shadow_color<sup>4</sup>: 阴影颜色。
* [节点注解](#节点注解)


### <a id="table1">InnerGlow</a>
生成内发光。
![image](image/inner_glow_example.png)

节点选项说明:   
![image](image/inner_glow_node.png)
* background_image<sup>1</sup>: 背景图像。
* layer_image<sup>1</sup>: 用于合成的层图像。
* layer_mask<sup>1,2</sup>: 层图像的遮罩，发光按此生成。
* invert_mask: 是否反转遮罩。
* blend_mode<sup>3</sup>: 发光的混合模式。
* opacity: 发光的不透明度。
* brightness: 发光亮度。
* glow_range: 发光范围。
* blur:发光模糊程度。
* light_color<sup>4</sup>: 发光中心颜色。
* glow_colo<sup>4</sup>: 辉光外围颜色。
* [节点注解](#节点注解)


### <a id="table1">Stroke</a>
生成描边。
![image](image/stroke_example.png)

节点选项说明:   
![image](image/stroke_node.png)
* background_image<sup>1</sup>: 背景图像。
* layer_image<sup>1</sup>: 用于合成的层图像。
* layer_mask<sup>1，2</sup>: 层图像的遮罩，描边按此生成。
* invert_mask: 是否反转遮罩。
* blend_mode<sup>3</sup>: 描边的混合模式。
* opacity: 不透明度。
* stroke_grow: 描边扩张/收缩幅度，正值是扩张，负值是收缩。
* stroke_width: 描边宽度。
* blur: 描边模糊。
* stroke_color<sup>4</sup>: 描边颜色。
* [节点注解](#节点注解)


### <a id="table1">GradientOverlay</a>
渐变覆盖
![image](image/gradient_overlay_example.png)

节点选项说明:   
![image](image/gradient_overlay_node.png)
* background_image<sup>1</sup>: 背景图像。
* layer_image<sup>1</sup>: 用于合成的层图像。
* layer_mask<sup>1，2</sup>: 层图像的遮罩。
* invert_mask: 是否反转遮罩。
* blend_mode<sup>3</sup>: 描边的混合模式。
* opacity: 不透明度。
* start_color: 渐变开始端的颜色。
* start_alpha: 渐变开始端的透明度。
* end_color: 渐变结束端的颜色。
* end_alpha: 渐变结束端的透明度。
* angle: 渐变旋转角度。
* [节点注解](#节点注解)


### <a id="table1">ColorOverlay</a>
颜色覆盖
![image](image/color_overlay_example.png)

节点选项说明:  
![image](image/color_overlay_node.png)
* background_image<sup>1</sup>: 背景图像。
* layer_image<sup>1</sup>: 用于合成的层图像。
* layer_mask<sup>1，2</sup>: 层图像的遮罩。
* invert_mask: 是否反转遮罩。
* blend_mode<sup>3</sup>: 描边的混合模式。
* opacity: 不透明度。
* color: 覆盖的颜色。
* [节点注解](#节点注解)

# <a id="table1">LayerColor</a>
![image](image/layercolor_title.png)
![image](image/layercolor_nodes.png)

### <a id="table1">LUT</a> Apply
将LUT应用到图像。仅支持.cube格式的LUT文件。

节点选项说明:  
![image](image/lut_apply_node.png)
* LUT<sup>*</sup>: 这里列出了LUT文件夹中可用的.cube文件列表，选中的LUT文件将被应用到图像。

<sup>*</sup>LUT文件夹在resource_dir.ini中定义，这个文件位于插件根目录下。用文本编辑软件打开，找到“LUT_dir=”开头的这一行，编辑“=”之后为自定义文件夹路径名。这个文件夹里面所有的.cube文件将在ComfyUI初始化时被收集并显示在节点的列表中。如果ini中设定的文件夹无效，将启用插件自带的LUT文件夹。

### <a id="table1">ColorAdapter</a>
自动调整图片色调，使之与参考图片相似。
![image](image/color_adapter_example.png)

节点选项说明:  
![image](image/color_adapter_node.png)
* opacity: 图像调整色调之后的不透明度。

### <a id="table1">Exposure</a>
改变图像的曝光。
![image](image/exposure_example.png)

节点选项说明:  
* exposure: 曝光值。更高的数值表示更亮的曝光。

### <a id="table1">Gamma</a>
改变图像的Gamma值。

节点选项说明:  
![image](image/gamma_node.png)
* gamma: 图像的Gamma值。

### <a id="table1">Brightness</a> & Contrast
改变图像的亮度、对比度和饱和度。

节点选项说明:  
![image](image/brightness_&_contrast_node.png)
* brightness: 图像的亮度。
* contrast: 图像的对比度。
* saturation: 图像的色彩饱和度。

### <a id="table1">RGB</a>
对图像的RGB各通道进行调整。

节点选项说明:  
![image](image/RGB_node.png)
* R: 图像的R通道。
* G: 图像的G通道。
* B: 图像的B通道。

### <a id="table1">YUV</a>
对图像的YUV各通道进行调整。

节点选项说明:  
![image](image/YUV_node.png)
* Y: 图像的Y通道。
* U: 图像的U通道。
* V: 图像的V通道。

### <a id="table1">LAB</a>
对图像的LAB各通道进行调整。

节点选项说明:  
![image](image/LAB_node.png)
* L: 图像的L通道。
* A: 图像的A通道。
* B: 图像的B通道。

### <a id="table1">HSV</a>
对图像的HSV各通道进行调整。

节点选项说明:  
![image](image/HSV_node.png)
* H: 图像的H通道。
* S: 图像的S通道。
* V: 图像的V通道。

# <a id="table1">LayerUtility</a>
![image](image/layerutility_nodes.png)


### <a id="table1">ImageBlendAdvance</a>
用于合成图层，允许在背景图片上合成与之不同尺寸的图层图片，并且设置位置和变换。提供多种混合模式供选择，可设置透明度。

节点提供了图层变换方法和抗锯齿选项。有助于提高合成画质。

节点提供了mask输出可用于后续工作流。
![image](image/image_blend_advance_example.png)

节点选项说明:   
![image](image/image_blend_advance_node.png)
* background_image: 背景图像。
* layer_image<sup>5</sup>: 用于合成的层图像。
* layer_mask<sup>2,5</sup>: 层图像的遮罩。
* invert_mask: 是否反转遮罩。
* blend_mode<sup>3</sup>: 图层混合模式。
* opacity: 不透明度。
* x_percent: 图层在背景图上的水平位置，用百分比表示，最左侧是0，最右侧是100，可以是小于0或者超过100，那表示图层有部分内容在画面之外。
* y_percent: 图层在背景图上的垂直位置，用百分比表示，最上侧是0，最下侧是100。例如设置为50表示垂直居中，20是偏上，80则是偏下。
* mirror: 镜像翻转。提供2种翻转模式, 水平翻转和垂直翻转。
* scale: 图层放大倍数，1.0 表示原大。
* aspect_ratio: 图层长宽比。1.0 是原始比例，大于此值表示拉长，小于此值表示压扁。
* rotate: 图层旋转度数。
* transform_method: 用于图层放大和旋转的采样方法，包括lanczos、bicubic、hamming、bilinear、box和nearest。不同的采样方法会影响合成的画质和画面处理时间。
* anti_aliasing: 抗锯齿，范围从0-16，数值越大，锯齿越不明显。过高的数值将显著降低节点的处理速度。
* [节点注解](#节点注解)

### <a id="table1">CropByMask</a>
将图片按照mask范围裁切，可设置四周边框保留大小。这个节点与[RestoreCropBox](#RestoreCropBox)和[ImageScaleRestore](#ImageScaleRestore)配合使用，可以对图片的局部进行裁切，放大修改后贴回原处。
![image](image/corp_by_mask_example.png)

节点选项说明:   
![image](image/corp_by_mask_node.png)
* image<sup>5</sup>: 输入的图像。
* mask_for_crop<sup>5</sup>: image的遮罩，将自动按照遮罩范围进行裁切。
* invert_mask: 是否反转遮罩。
* detect: 探测方法，min_bounding_rect是最小外接矩形, max_inscribed_rect是最大内接矩形。
* top_reserve: 裁切顶端保留大小。
* bottom_reserve: 裁切底部保留大小。
* left_reserve: 裁切左侧保留大小。
* right_reserve: 裁切右侧保留大小。
* [节点注解](#节点注解)

输出:
* croped_image: 裁切后的图片。
* croped_mask: 裁切后的遮罩。
* crop_box: 裁切box数据，在RestoreCropBox节点恢复时使用。
* box_preview: 裁切位置预览图，红色是探测到的范围，绿色是加上保留边框后裁切的范围。

### <a id="table1">RestoreCropBox</a>
将被[CropByMask](#CropByMask)裁切后的图片恢复到原图。

节点选项说明:   
![image](image/restore_crop_box_node.png)
* background_image: 裁切前的原图。
* croped_image<sup>5</sup>: 裁切后的图片。如果中间经过放大处理，恢复前需将尺寸还原。
* croped_mask<sup>2,5</sup>: 裁切后的遮罩。
* crop_box: 裁切时的box数据。
* invert_mask: 是否反转遮罩。
* [节点注解](#节点注解)

### <a id="table1">ImageScaleRestore</a>
图像缩放。此节点成对使用时，在第二个节点可自动还原图像到原始大小。
![image](image/image_scale_restore_example.png)

节点选项说明:   
![image](image/image_scale_restore_node.png)
* image<sup>5</sup>: 输入的图像。
* mask<sup>2,5</sup>: 图像的遮罩。
* original_size: 可选输入，用于恢复图片到原始大小。
* scale: 缩放比例。当有original_size输入，或者scale_by_longest_side设置为True时，此项设置将被忽略。
* scale_by_longest_side: 允许按长边尺寸缩放。
* longest_side: scale_by_longest_side被设置为True时，此项将作为是图像长边的长度。当有original_size输入时，此项设置将被忽略。
* [节点注解](#节点注解)

输出:
* image: 缩放后的图像。
* mask: 如果有mask输入，将输出缩放后的mask。
* original_size: 图像的原始大小数据，用于后续节点进行恢复。

### <a id="table1">ImageMaskScaleAs</a>
将图像或遮罩缩放到参考图像（或遮罩）的大小。
![image](image/image_mask_scale_as_example.png)

节点选项说明:   
![image](image/image_mask_scale_as_node.png)
* scale_as<sup>*</sup>: 参考大小。可以是图像image，也可以是遮罩mask。
* image: 待缩放的图像。此选项为可选输入，如果没有输入将输出纯黑图片。
* mask: 待缩放的遮罩。此选项为可选输入，如果没有输入将输出纯黑遮罩。
* fit: 缩放画幅宽高比模式。当原图与缩放尺寸画幅宽高比例不一致时，有3种模式可以选择, letterbox模式保留完整的画幅，空白处用黑色补足；crop模式保留完整的短边，长边超出部分将被切除；fill模式不保持画幅比例，宽高各自填满画面。
* method: 缩放的采样方法，包括lanczos、bicubic、hamming、bilinear、box和nearest。

<sup>*</sup>仅限输入image和mask, 如果强制接入其他类型输入，将导致节点错误。

输出:
* image: 如果有image输入，将输出缩放后的图像。
* mask: 如果有mask输入，将输出缩放后的遮罩。
* original_size: 图像的原始大小数据，用于后续节点进行恢复。

### <a id="table1">ImageShift</a>
使图片产生位移。此节点支持位移接缝遮罩的输出，方便制作连续贴图。
![image](image/image_shift_example.png)

节点选项说明:   
![image](image/image_shift_node.png)
* image<sup>5</sup>: 输入的图像。
* mask<sup>2,5</sup>: 图像的遮罩。
* shift_x: 位移的横向距离。
* shift_y: 位移的纵向距离。
* cyclic: 位移出界的部分是否循环。
* background_color: 背景颜色。如果cyclic设置为False,将使用这里的设置作为背景颜色。
* border_mask_width: 接缝遮罩宽度。
* border_mask_blur: 接缝遮罩模糊。
* [节点注解](#节点注解)

### <a id="table1">ImageBlend</a>
一个用于合成图层的简单节点，提供多种混合模式供选择，可设置透明度。
![image](image/image_blend_example.png)

节点选项说明:   
![image](image/image_blend_node.png)
* background_image<sup>1</sup>: 背景图像。
* layer_image<sup>1</sup>: 用于合成的层图像。
* layer_mask<sup>1,2</sup>: 层图像的遮罩。
* invert_mask: 是否反转遮罩。
* blend_mode<sup>3</sup>: 图层混合模式。
* opacity: 不透明度。
* [节点注解](#节点注解)

### <a id="table1">ImageOpacity</a>
调整图像不透明度。
![image](image/image_opacity_example.png)

节点选项说明:   
* image<sup>5</sup>: 图像输入，支持RGB和RGBA输入。
* mask<sup>2,5</sup>: 遮罩输入。
* invert_mask: 是否反转遮罩。
* opacity: 不透明度。
* [节点注解](#节点注解)

### <a id="table1">ColorPicker</a>
在色板上选取颜色并输出。
改自[mtb nodes](https://github.com/melMass/comfy_mtb)的web extensions。
![image](image/color_picker.png)

节点选项说明:
* mode： 输出格式，可选十六进制(HEX)或十进制(DEC)。

输出:
* value: 字符串格式。

### <a id="table1">GetColorTone</a>
从图片中获取主颜色或平均色。
![image](image/get_color_tone_example.png)

节点选项说明:
![image](image/get_color_tone_node.png)
* mode： 模式，有两种可选择，主颜色main_color和平均色average。

输出:
* RGB color in HEX: 使用16进制RGB字符串格式描述，例如 '#FA3D86'。
* HSV color in list: HSV颜色值，使用list格式描述。

### <a id="table1">ExtendCanvas</a>
扩展画布。
![image](image/extend_canvas_example.png)

节点选项说明:
![image](image/extend_canvas_node.png)
* invert_mask: 是否反转遮罩。
* top: 顶部扩展值。
* bottom: 底部扩展值。
* left: 左侧扩展值。
* right: 右侧扩展值。
* color; 画布颜色

### XY to <a id="table1">Percent</a>
![image](image/xy2percent_example.png)
将绝对坐标转换为百分比坐标。

![image](image/xy2percent_node.png)
节点选项说明:
* x: 坐标x值。
* y: 坐标y值。

### <a id="table1">ColorImage</a>
![image](image/color_image_example.png)
生成一张指定颜色和大小的图片。

![image](image/color_image_node.png)
节点选项说明:
* width: 图像宽度。
* height: 图像高度。
* color<sup>4</sup>: 颜色。

### <a id="table1">GradientImage</a>
![image](image/gradient_image_example.png)
生成一张指定大小和指定颜色渐变的图片。

![image](image/gradient_image_node.png)
节点选项说明:
* width: 图像宽度。
* height: 图像高度。
* angle: 渐变角度。
* start_color<sup>4</sup>: 开始端颜色。
* end_color<sup>4</sup>: 结束端颜色。

### <a id="table1">TextImage</a>
![image](image/text_image_example.png)
从文字生成图片以及遮罩。支持字间距行间距调整，横排竖排调整，可设置文字的随机变化，包括大小和位置的随机变化。

![image](image/text_image_node.png)
节点选项说明:
* size_as<sup>*</sup>: 此处输入图像或遮罩，将按照其尺寸生成输出图像和遮罩。注意，此输入优先级高于下面的width和height。
* font_file<sup>**</sup>: 这里列出了font文件夹中可用的字体文件列表，选中的字体文件将被用来生成图像。
* spacing: 字间距,以像素为单位。
* leading: 行间距,以像素为单位。
* horizontal_border: 侧边边距。此处数值表示的是百分比，例如50表示起点位于两侧的正中央。如果文字是横排，是左侧边距，竖排则是右侧边距。
* vertical_border: 顶部边距。此处数值表示的是百分比，例如10表示起点位于距顶部10%的位置。
* scale: 文字总体大小。文字的初始大小是根据画面尺寸和文字内容自动计算，默认以最长的行或者列适配画面宽或者高。调整此处数值将整体放大和缩小文字。此处数值表示的是百分比，例如60表示缩放到60%。
* variation_range: 字符随机变化范围。此数值大于0时，字符将产生大小和位置的随机变化，数值越大，变化幅度越大。
* variation_seed: 随机变化的种子。固定此数值，每次产生的单个文字的变化不会改变。
* layout: 文字排版。有横排和竖排可选择。
* width; 画面的宽度。如果有size_as输入，此设置将被忽略。
* height; 画面的高度。如果有size_as输入，此设置将被忽略。
* text_color: 文字颜色。
* background_color: 背景颜色。

<sup>*</sup>仅限输入image和mask, 如果强制接入其他类型输入，将导致节点错误。

<sup>**</sup>font文件夹在resource_dir.ini中定义，这个文件位于插件根目录下。用文本编辑软件打开，找到“FONT_dir=”开头的这一行，编辑“=”之后为自定义文件夹路径名。这个文件夹里面所有的.ttf和.otf文件将在ComfyUI初始化时被收集并显示在节点的列表中。如果ini中设定的文件夹无效，将启用插件自带的font文件夹。


### <a id="table1">ImageChannelSplit</a>
![image](image/image_channel_split_example.png)
将图像通道拆分为单独的图片。

节点选项说明:
![image](image/image_channel_split_node.png)
* mode: 通道模式。包含RGBA, YCbCr, LAB和HSV。

### <a id="table1">ImageChannelMerge</a>
![image](image/image_channel_merge_example.png)
将各通道合并为一张图片。

节点选项说明:
![image](image/image_channel_merge_node.png)
* mode: 通道模式。包含RGBA, YCbCr, LAB和HSV。

### <a id="table1">GetImageSize</a>
![image](image/get_image_size_node.png)
获取图片的宽度和高度。

输出:
* width: 图像宽度。
* height: 图像高度。
* original_size: 图像的原始大小数据，用于后续节点进行恢复。

### <a id="table1">PrintInfo</a>
![image](image/print_info_node.png)
用于给工作流调试提供辅助。当运行时，任何接上这个节点的对象的属性将被打印到控制台。

这个节点允许任意类型的输入。



# <a id="table1">LayerMask</a>
![image](image/layermask_nodes.png)

### <a id="table1">MaskBoxDetect</a>
探测mask所在区域，并输出位置和大小。
![image](image/mask_box_detect_example.png)

节点选项说明:  
![image](image/mask_box_detect_node.png)
* detect: 探测方法，min_bounding_rect是最小外接矩形, max_inscribed_rect是最大内接矩形。
* x_adjust: 修正探测之后的水平偏移。
* y_adjust: 修正探测之后的垂直偏移。
* scale_adjust: 修正探测之后的缩放偏移。

输出：
* box_preview: 探测结果预览图。红色表示探测到的结果，绿色表示加上修正后的输出结果。
* x_percent: 水平位置以百分比输出。
* y_percent: 垂直位置以百分比输出。
* width: 宽度输出。
* height: 高度输出。
* x: 左上角位置x坐标输出。
* y: 左上角位置y坐标输出。

### <a id="table1">SegmentAnythingUltra</a>
对[ComfyUI Segment Anything](https://github.com/storyicon/comfyui_segment_anything)的改进，结合了spacepxl的[ComfyUI-Image-Filters](https://github.com/spacepxl/ComfyUI-Image-Filters)的Alpha Matte节点，使遮罩有更具细节的边缘。

*请参照ComfyUI Segment Anything的安装方法安装模型。如果已经正确安装了ComfyUI Segment Anything，可跳过此步骤。
* 从 [这里](https://huggingface.co/bert-base-uncased/tree/main) 下载 config.json，model.safetensors，tokenizer_config.json，tokenizer.json 和 vocab.txt 5个文件到 ComfyUI/models/bert-base-uncased文件夹。
* 下载 [GroundingDINO_SwinT_OGC config file](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py), [GroundingDINO_SwinT_OGC model](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth), 
[GroundingDINO_SwinB config file](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py), [GroundingDINO_SwinB model](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth) 到 ComfyUI/models/grounding-dino文件夹。
* 下载 [sam_vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)，[sam_vit_l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth), 
[sam_vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth), [sam_hq_vit_h](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth),
[sam_hq_vit_l](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth), [sam_hq_vit_b](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth), 
[mobile_sam](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt) 这几个文件到ComfyUI/models/sams文件夹。

![image](image/segment_anything_ultra_compare.png)
![image](image/segment_anything_ultra_example.png)

节点选项说明:  
![image](image/segment_anything_ultra_node.png)
* sam_model: 选择SAM模型。
* ground_dino_model: 选择Grounding DINO模型。
* threshold: SAM阈值。
* detail_range: 边缘细节范围。
* black_point: 边缘黑色采样阈值。
* white_point: 边缘黑色采样阈值。
* process_detail: 此处设为False将跳过边缘处理以节省运行时间。
* prompt: SAM的prompt输入。


### <a id="table1">RemBgUltra</a>
去除背景。与类似的背景移除节点相比，这个节点具有超高的边缘细节。
本节点结合了spacepxl的[ComfyUI-Image-Filters](https://github.com/spacepxl/ComfyUI-Image-Filters)的Alpha Matte节点，以及ZHO-ZHO-ZHO的[ComfyUI-BRIA_AI-RMBG](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG)的功能。
*将[BRIA Background Removal v1.4](https://huggingface.co/briaai/RMBG-1.4)模型文件(model.pth)下载至/custom_nodes/ComfyUI_LayerStyle/RMBG-1.4。
该模型由 BRIA AI 开发，可作为非商业用途的开源模型。

![image](image/rembg_ultra_example.png)

节点选项说明:  
![image](image/rembg_ultra_node.png)
* detail_range: 边缘细节范围。
* black_point: 边缘黑色采样阈值。
* white_point: 边缘黑色采样阈值。
* process_detail: 此处设为False将跳过边缘处理以节省运行时间。

### <a id="table1">PixelSpread</a>
对图像的遮罩边缘部分进行像素扩张预处理，可有效改善图像合成的边缘。
![image](image/pixel_spread_example.png)

节点选项说明:  
![image](image/pixel_spread_node.png)
* invert_mask: 是否反转遮罩。
* mask_grow: 遮罩扩张幅度。

### <a id="table1">MaskByDifferent</a>
计算两张图像不同之处，并输出为遮罩。
![image](image/mask_by_different_example.png)

节点选项说明:  
![image](image/mask_by_different_node.png)
* gain: 计算增益。调高此值，微弱的差异将更显著的呈现。
* fix_gap: 修补遮罩内部缝隙。更高的值将修补更大的缝隙。
* fix_threshold: 修补阈值。
* main_subject_detect: 此项设为True将开启主体侦测，忽略主体之外的差异。


### <a id="table1">MaskEdgeUltraDetail</a>
处理较粗糙的遮罩使其获得超精细边缘。
本节点结合了spacepxl的[ComfyUI-Image-Filters](https://github.com/spacepxl/ComfyUI-Image-Filters)的Alpha Matte和Guided Filter Alpha两者功能。
![image](image/mask_edge_ultra_detail_example.png)

节点选项说明:  
![image](image/mask_edge_ultra_detail_node.png)
* method: 提供PyMatting和OpenCV-GuidedFilter两种方法处理边缘。PyMatting处理速度较慢，但是对于视频，建议使用这种方法获得更平滑的遮罩序列。
* mask_grow: 遮罩扩张幅度。正值是向外扩张，负值是向内收缩。对于较粗糙的遮罩，通常使用负值使其边缘收缩以获得更好的效果。
* fix_gap: 修补遮罩中的空隙。如果遮罩中有比较明显的空隙，适当调高此数值。
* fix_threshold: 修补遮罩的阈值。
* detail_range: 边缘细节范围。
* black_point: 边缘黑色采样阈值。
* white_point: 边缘黑色采样阈值。


### <a id="table1">MaskGrow</a>
对mask进行扩张收缩边缘和模糊处理
![image](image/mask_grow_example.png)

节点选项说明:  
![image](image/mask_grow_node.png)
* invert_mask: 是否反转遮罩。
* grow: 扩张幅度。正值是向外扩张，负值是向内收缩。
* blur: 模糊。

### <a id="table1">MaskEdgeShrink</a>
使mask边缘平滑地过渡收缩，并保留边缘细节。
![image](image/mask_edge_shrink_example.png)

节点选项说明:  
![image](image/mask_edge_shrink_node.png)
* invert_mask: 是否反转遮罩。
* shrink_level: 收缩平滑级别。
* soft: 平滑幅度。
* edge_shrink: 边缘收缩幅度。
* edge_reserve: 保留边缘细节幅度, 100为完全保留，0为完全不保留。

MaskGrow与MaskEdgeShrink效果对比
![image](image/mask_edge_compare.png)

### <a id="table1">MaskMotionBlur</a>
使mask产生运动模糊。
![image](image/mask_motion_blur_example.png)

节点选项说明:  
![image](image/mask_motion_blur_node.png)
* invert_mask: 是否反转遮罩。
* blur: 模糊大小。
* angle: 模糊角度。

### <a id="table1">MaskGradient</a>
使mask从一侧产生渐变。
![image](image/mask_gradient_example.png)

节点选项说明:  
![image](image/mask_gradient_node.png)
* invert_mask: 是否反转遮罩。
* gradient_side: 从哪个边产生渐变。有四个方向：顶侧top、底侧bottom、左侧left、右侧right。
* gradient_scale: 渐变距离。默认值100表示渐变产生一侧完全透明，另一侧完全不透明。数值越小，从透明到不透明的距离越短。
* gradient_offset: 渐变位置偏移。
* opacity: 渐变的不透明度。

### <a id="table1">MaskStroke</a>
产生mask轮廓描边。
![image](image/mask_stroke_example.png)

节点选项说明:  
![image](image/mask_stroke_node.png)
* invert_mask: 是否反转遮罩。
* stroke_grow: 描边扩张/收缩幅度，正值是扩张，负值是收缩。
* stroke_width: 描边宽度。
* blur: 描边模糊。

### <a id="table1">MaskPreview</a>
预览mask
![image](image/mask_invert.png)


### <a id="table1">MaskInvert</a>
mask反转
![image](image/mask_invert_node.png)



# <a id="table1">LayerFilter</a>
![image](image/layerfilter_nodes.png)

### <a id="table1">Sharp</a> & Soft
为图像增强细节或抹平细节。
![image](image/sharp_and_soft_example.png)

节点选项说明:
![image](image/sharp_and_soft_node.png)
* enhance: 提供四个预设档位，分别是very sharp、sharp、soft和very soft。

### <a id="table1">SkinBeauty</a>
磨皮效果。
![image](image/skin_beauty_example.png)

节点选项说明:
![image](image/skin_beauty_node.png)
* smooth: 皮肤平滑度。
* threshold: 磨皮范围。数值越小范围越大。
* opacity: 磨皮的不透明度。


### <a id="table1">WaterColor</a>
水彩画效果。
![image](image/water_color_example.png)

节点选项说明:
![image](image/water_color_node.png)
* line_density: 线条密度。
* opacity: 水彩效果的不透明度。


### <a id="table1">SoftLight</a>
柔光效果。
![image](image/soft_light_example.png)

节点选项说明:
![image](image/soft_light_node.png)
* soft: 柔光大小。
* threshold: 柔光范围。柔光从画面最明亮的部分呈现。数值越低范围越大，越高范围越小。
* opacity: 柔光的不透明度。

### <a id="table1">ChannelShake</a>
通道错位。类似抖音logo的效果。
![image](image/channel_shake_example.png)

节点选项说明:
![image](image/channel_shake_node.png)
* distance: 通道分离的距离。
* angle: 通道分离的角度。
* mode: 通道错位排列顺序。

### <a id="table1">ColorMap</a>
伪彩色热力图效果。
![image](image/colormap_result.png)

节点选项说明:
![image](image/color_map_node.png)
* color_map: 效果类型。共22种，各种类型的效果如上图所示。
* opacity: 伪彩色效果的不透明度。

### <a id="table1">MotionBlur</a>
运动模糊。
![image](image/motion_blur_example.png)

节点选项说明:
* angle: 模糊角度。
* blur: 模糊大小。

### <a id="table1">GaussianBlur</a>
高斯模糊。
![image](image/gaussian_blur_example.png)

节点选项说明:
* blur: 模糊大小。

## <a id="table1">节点注解</a>
<sup>1</sup>  image、mask和background_image(如果有输入)这三项必须是相同的尺寸。
<sup>2</sup>  mask不是必须的输入项，默认使用image的alpha通道，如果image输入不包含alpha通道将自动创建整个图像的alpha通道。如果输入mask，原本的alpha通道将被mask覆盖。
<sup>3</sup>   <a id="table1">混合模式</a>包括normal、multply、screen、add、subtract、difference、darker、lighter、color_burn、color_dodge、linear_burn、linear_dodge、overlay、soft_light、hard_light、vivid_light、pin_light、linear_light、hard_mix, 共19种混合模式
<sup>4</sup>  颜色使用16进制RGB字符串格式描述，例如 '#FA3D86'。
<sup>5</sup>  image和mask这两项必须是相同的尺寸。
<sup>3</sup> 混合模式预览
![image](image/blend_mode_result.png)

## 使用示例

在workflow目录下有json格式的工作流示例文件。


## 安装方法

* 推荐使用 ComfyUI Manager 安装。
* 或者在CompyUI插件目录(例如“CompyUI\custom_nodes\”)中打开cmd窗口，键入```git clone https://github.com/chflame163/ComfyUI_LayerStyle.git```安装。或者下载解压zip文件，将得到的文件夹复制到 ComfyUI\custom_nodes\
* 安装依赖包，在资源管理器ComfyUI\custom_nodes\ComfyUI_WordCloud 插件目录位置打开cmd窗口，输入以下命令：  
```..\..\..\python_embeded\python.exe -m pip install -r requirements.txt```
* 重新打开ComfyUI。
