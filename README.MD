# ComfyUI Layer Style
A set of nodes for ComfyUI it generate image like Adobe Photoshop's Layer Style.
the Drop Shadow is first completed node, and  follow-up work is in progress.

[中文说明点这里](./README_CN.MD)

## Node Description

### Drop Shadow:
Generate shadow
![image](image/drop_shadow_node.png)


Node options：   
* background_image: The background image.
* layer_image: Layer images for composite.
* layer_mask: Mask for layer_image, shadows are generated according to their shape.
* invert_mask: Whether to reverse the mask.
* blend_mode: Blending method of shadows, there are **_normal, multply, screen, add, subtract, difference, darker_** and **_lighter_**.
* opacity: Opacity of shadow.
* distance_x: Horizontal offset of shadow.
* distance_y: Vertical offset of shadow.
* grow: Shadow expansion amplitude.
* blur：Shadow blur level.
* shadow_color: Shadow color, described in hexadecimal RGB format.

Output type：
* image：Completed image.
* shadow_mask：Shadow's mask.

## Example workflow
![image](image/drop_shadow.png)
image Some JSON workflow files in the workflow directory, that is example for ComfyUI.

## How to install 

* Open the cmd window in the plugin directory of ComfyUI, like "ComfyUI\custom_nodes\"，type```git clone https://github.com/chflame163/ComfyUI_WordCloud.git```
or download the zip file and extracted, copy the resulting folder to ComfyUI\custom_ Nodes\

* Install dependency packages, open the cmd window in the WordCloud plugin directory like "ComfyUI\custom_ Nodes\ComfyUI_WordCloud" and enter the following command:
```..\..\..\python_embeded\python.exe -m pip install -r requirements.txt```

* Restart ComfyUI