## 项目简介

这个项目旨在生成合成图像并进行对象检测或其他计算机视觉任务。通过将图像从输入文件夹中随机选择并将其粘贴到画布上生成合成图像。生成的图像和相应的注释文件将保存在输出文件夹中。项目包含了三个主要的Python脚本：

- `synthetic_image_generator.py`：该脚本用于生成合成图像。它从背景图像开始，然后根据指定的模式和参数，选择图像并将它们粘贴到画布上。同时，生成相应的注释文件。

- `synthetic_image_generator_parallel.py`：与`synthetic_image_generator.py`类似，但使用了多进程并行处理，以提高生成速度。

- `draw_bounding_boxes.py`：该脚本用于从XML文件中读取边界框信息，并在图像上绘制边界框和标签。

你可以根据项目的需求和具体情况，进一步修改和补充README文件的内容。
## 文件结构
- README.md
- synthetic_image_generator.py
- synthetic_image_generator_parallel.py
- draw_bounding_boxes.py
- /output
    - output_1.jpg
    - output_1.xml
    - output_2.jpg
    - output_2.xml
    ...
- /JPEGImages
    - image1.jpg
    - image2.jpg
    ...
- /Annotations
    - annotation1.xml
    - annotation2.xml
    ...
## 功能特点
- 可以生成合成图像

- 支持对象检测任务

- 多进程并行处理（适用于`synthetic_image_generator_parallel.py`）

- 绘制边界框和标签（适用于`draw_bounding_boxes.py`
