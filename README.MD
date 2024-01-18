# ComfyUI Layer Style
A set of nodes for ComfyUI that generate image like Adobe Photoshop's Layer Style.


Currently completed Drop Shadow and OuterGlow nodes, and follow-up work is in progress.

[中文说明点这里](./README_CN.MD)

## Update
Remove the mask channel output from the DropShadow node; Add the OuterGlow node.

## Node Description

### Drop Shadow:
Generate shadow
![image](image/drop_shadow_example.png)



Node options:
![image](image/drop_shadow_node.png)
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1</sup>: Mask for layer_image, shadows are generated according to their shape.
* invert_mask: Whether to reverse the mask.
* blend_mode: Blending method of shadows, there are **_normal, multply, screen, add, subtract, difference, darker_** and **_lighter_**.
* opacity: Opacity of shadow.
* distance_x: Horizontal offset of shadow.
* distance_y: Vertical offset of shadow.
* grow: Shadow expansion amplitude.
* blur：Shadow blur level.
* shadow_color: Shadow color, described in hexadecimal RGB format.
 
<sup>1</sup>  The layer_image, layer_mask and the background_image, These three items must be of the same size.

Output type：
* image


### OuterGlow：
Generate outer glow
![image](image/outer_glow_example.png)

Node options:
![image](image/outer_glow_node.png)
* background_image<sup>1</sup>: The background image.
* layer_image<sup>1</sup>: Layer image for composite.
* layer_mask<sup>1</sup>: Mask for layer_image, shadows are generated according to their shape.
* invert_mask: Whether to reverse the mask.
* blend_mode: Blending method of glow, there are **_normal, multply, screen, add, subtract, difference, darker_** and **_lighter_**.
* opacity: Opacity of Glow.
* grow: Glow expansion amplitude.
* blur：Gkow blur level.
* glow_color:  Glow color, described in hexadecimal RGB format.

<sup>1</sup>  The layer_image, layer_mask and the background_image, These three items must be of the same size.

输出：
* image

## Example workflow
image Some JSON workflow files in the workflow directory, that is example for ComfyUI.

## How to install 

* Open the cmd window in the plugin directory of ComfyUI, like "ComfyUI\custom_nodes\"，type```git clone https://github.com/chflame163/ComfyUI_LayerStyle.git```
or download the zip file and extracted, copy the resulting folder to ComfyUI\custom_ Nodes\

* Install dependency packages, open the cmd window in the WordCloud plugin directory like "ComfyUI\custom_ Nodes\ComfyUI_WordCloud" and enter the following command:
```..\..\..\python_embeded\python.exe -m pip install -r requirements.txt```

* Restart ComfyUI