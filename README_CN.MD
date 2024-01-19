# ComfyUI Layer Style：
ComfyUI插件，通过一组节点实现仿照Adobe Photoshop的图层样式。

当前已完成Drop Shadow和OuterGlow节点和一些辅助节点，后续工作进行中。

## 更新说明：
* 添加MaskInvert节点。
* 添加Stroke节点。
* 添加MaskPreview节点。
* 添加ImageOpacity节点。
* layer_mask修改为非必选, 默认使用layer_image的alpha通道，允许通过输入mask改变之，但是尺寸必须一致。
* 添加ImageBlend节点。
* 添加OuterGlow节点。
* 首个节点DropShadow提交。

## 节点说明：

### DropShadow:
生成阴影。
![image](image/drop_shadow_example.png)


节点选项说明：   
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
* blur：阴影模糊程度。
* shadow_color: 阴影颜色，使用16进制RGB格式描述。


输出：
* image

### OuterGlow：
生成外发光。
![image](image/outer_glow_example.png)

节点选项说明：   
![image](image/outer_glow_node.png)
* background_image<sup>1</sup>: 背景图像。
* layer_image<sup>1</sup>: 用于合成的层图像。
* layer_mask<sup>1,2</sup>: 层图像的遮罩，外发光按此生成。
* invert_mask: 是否反转遮罩。
* blend_mode<sup>3</sup>: 发光的混合模式。
* opacity: 发光的不透明度。
* grow: 发光扩张幅度。
* blur：发光模糊程度。
* glow_color: 发光颜色，使用16进制RGB格式描述。


输出：
* image

### Stroke：
生成描边。
![image](image/stroke_example.png)

节点选项说明：   
![image](image/stroke_node.png)
* background_image<sup>1</sup>: 背景图像。
* layer_image<sup>1</sup>: 用于合成的层图像。
* layer_mask<sup>1，2</sup>: 层图像的遮罩，描边按此生成。
* invert_mask: 是否反转遮罩。
* blend_mode<sup>3</sup>: 描边的混合模式。
* opacity: 不透明度。
* shrink<sup>*</sup>: 描边收缩幅度。
* expansion<sup>*</sup>：描边扩张幅度。
* stroke_color: 描边颜色，使用16进制RGB格式描述。

<sup>*</sup>描边的宽度等于收缩值加上扩张值。两个值分开设置能更方便的调整边框范围。


输出：
* image

### ImageBlend：
用于合成图层，提供多种混合模式供选择，可设置透明度。
![image](image/image_blend_example.png)

节点选项说明：   
![image](image/image_blend_node.png)
* background_image<sup>1</sup>: 背景图像。
* layer_image<sup>1</sup>: 用于合成的层图像。
* layer_mask<sup>1,2</sup>: 层图像的遮罩。
* invert_mask: 是否反转遮罩。
* blend_mode<sup>3</sup>: 图层混合模式。
* opacity: 不透明度。


输出：
* image

### ImageOpacity:
调整图像不透明度。
![image](image/image_opacity_example.png)

节点选项说明：   
* image: 图像输入，支持RGB和RGBA输入。
* mask: 遮罩输入。
* invert_mask: 是否反转遮罩。
* opacity: 不透明度。

输出：
* image(RGBA)
* mask

### MaskPreview:
预览mask
![image](image/mask_preview_example.png)

输入：
* mask

### MaskInvert:
mask反转
![image](image/mask_invert.png)

输入：
* mask

输出： 
* mask


### ColorPicker:
在色板上选取颜色并输出。
改自[mtb nodes](https://github.com/melMass/comfy_mtb)的web extensions。
![image](image/color_picker.png)

节点选项说明：
* mode： 输出格式，可选十六进制(HEX)或十进制(DEC)。

输出： 
* string



### 节点选项注解：
<sup>1</sup>  image、mask和background_image(如果有输入)这三项必须是相同的尺寸。
<sup>2</sup>  mask不是必须的输入项，默认使用image的alpha通道，如果image输入不包含alpha通道将自动创建整个图像的alpha通道。如果输入mask，原本的alpha通道将被mask覆盖。
<sup>3</sup>: 混合模式包括normal、multply、screen、add、subtract、difference、darker和lighter。

## 使用示例：

在workflow目录下有json格式的工作流示例文件。

## 以下工作在计划列表中
### Layer Style节点：
* Inner Shadow
* Inner Glow
* Stroke
* Texture
* Color Overlay
* Gradient Overlay
* Neon
### 辅助节点：
* invert mask
* invert image
* load image+
* mask soft crop
* mask motion blur
* mask radial blur


## 安装方法：  

* 在CompyUI插件目录(例如“CompyUI\custom_nodes\”)中打开cmd窗口，键入```git clone https://github.com/chflame163/ComfyUI_LayerStyle.git```安装。或者下载解压zip文件，将得到的文件夹复制到 ComfyUI\custom_nodes\
* 安装依赖包，在资源管理器ComfyUI\custom_nodes\ComfyUI_WordCloud 插件目录位置打开cmd窗口，输入以下命令：  
```..\..\..\python_embeded\python.exe -m pip install -r requirements.txt```
* 重新打开ComfyUI。
