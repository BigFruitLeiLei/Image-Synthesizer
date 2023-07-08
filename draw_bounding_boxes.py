import os
import cv2
import xml.etree.ElementTree as ET

def draw_bounding_boxes(image_path, xml_filename):
    """
    从XML文件中读取边界框信息，并在图像上绘制边界框和标签。
    
    参数：
        - image_path：图像文件的路径
        - xml_filename：XML文件的文件名
    
    """

    # 获取当前脚本文件的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建图像文件的完整路径
    image_path = os.path.join(script_dir, image_path)

    # 构建XML文件的完整路径
    xml_path = os.path.join(script_dir, xml_filename)

    # 读取图像
    image = cv2.imread(image_path)

    # 检查图像是否成功加载
    if image is None:
        print(f"无法加载图像：{image_path}")
        return

    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 遍历每个对象
    for obj in root.findall('object'):
        # 获取边界框坐标
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        # 获取标签名称
        label = obj.find('name').text

        # 在图像上绘制边界框
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 在边界框左上角绘制标签
        cv2.putText(image, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例用法

image_filename = 'output_2.jpg'  # 图像文件名
xml_filename = 'output_2.xml'  # XML文件名
draw_bounding_boxes(image_filename, xml_filename)




