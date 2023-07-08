import os
import cv2
import random
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from tqdm import tqdm  # 添加进度条库

# 定义输入和输出文件夹，以及画布大小和目标物体大小
input_folder = "./JPEGImages"  
annotations_folder = "./Annotations" 
output_folder = "./output" 
canvas_width = 1920 
canvas_height = 1080 
min_target_size = 50 
max_target_size = 300 
mode = 1


def parse_annotation(annotation_path):
    """
    解析XML注释文件，提取目标物体的名称和边界框信息。

    参数：
        annotation_path (str): 注释文件的路径。

    返回：
        objects (list): 包含目标物体信息的列表，每个目标物体由名称和边界框坐标组成。
                        [(name, xmin, ymin, xmax, ymax), ...]
    """
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        objects = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            objects.append((name, xmin, ymin, xmax, ymax))
        return objects
    except Exception as e:
        print(f"Error reading annotation file {annotation_path}: {e}")
        return None


def is_overlapping(new_x, new_y, new_width, new_height, pasted_images):
    """
    检查新的图像区域是否与已经粘贴的图像区域重叠。

    参数：
        new_x (int): 新图像的左上角 x 坐标。
        new_y (int): 新图像的左上角 y 坐标。
        new_width (int): 新图像的宽度。
        new_height (int): 新图像的高度。
        pasted_images (list): 已经粘贴的图像的列表，每个图像由左上角坐标和宽度高度组成。
                              [(x, y, width, height), ...]

    返回：
        overlapping (bool): 如果新的图像与已经粘贴的图像重叠，则为 True，否则为 False。
    """
    for x, y, width, height in pasted_images:
        if not (new_x + new_width < x or new_x > x + width or new_y + new_height < y or new_y > y + height):
            return True
    return False


def create_images(filename,num_images, num_targets=15, max_attempts=50):
    """
    创建合成图像并生成相应的注释文件。

    参数：
        num_images (int): 要创建的合成图像数量。
        num_targets (int): 每个图像中的目标物体数量。
        max_attempts (int): 寻找非重叠位置的最大尝试次数。
    """
    background = cv2.imread(filename)
    #将图像从BGR转换为RGB
    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    
    for i in tqdm(range(num_images)):
        # 创建画布
        if mode == 0:
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        elif mode == 1:
            canvas = background_rgb.copy()
            canvas_height,canvas_width,channel = canvas.shape
            
        # 获取输入文件夹中的图像文件列表
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]

        # 创建根节点
        root = ET.Element("annotation")
        # 添加子节点
        ET.SubElement(root, "folder").text = "JPEGImages"
        ET.SubElement(root, "filename").text = f"output_{i+1}.jpg"
        ET.SubElement(root, "path").text = os.path.join(os.path.abspath(output_folder), f"output_{i+1}.jpg")
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "Unknown"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(canvas_width)
        ET.SubElement(size, "height").text = str(canvas_height)
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"

        pasted_images = []

        for _ in range(min(num_targets, len(image_files))):
            # 随机选择一个图像文件
            image_file = random.choice(image_files)
            try:
                # 读取图像
                image = cv2.imread(os.path.join(input_folder, image_file))
                if image is None:
                    continue
            except Exception as e:
                continue

            # 计算目标物体的尺寸
            image_height, image_width = image.shape[:2]
            target_size = random.randint(min_target_size, max_target_size)
            scaling_factor = target_size / max(image_height, image_width)
            # 调整图像大小
            resized_image = cv2.resize(image, (int(image_width * scaling_factor), int(image_height * scaling_factor)))

            new_width, new_height = resized_image.shape[1], resized_image.shape[0]

            for attempt in range(max_attempts):
                # 随机选择新图像的位置
                new_x = random.randint(0, canvas_width - new_width)
                new_y = random.randint(0, canvas_height - new_height)

                # 检查新图像是否与已经粘贴的图像重叠
                if not is_overlapping(new_x, new_y, new_width, new_height, pasted_images):
                    break
            else:
                print(f"Could not find a non-overlapping position for {image_file} after {max_attempts} attempts.")
                continue

            # 将图像粘贴到画布上
            canvas[new_y:new_y + new_height, new_x:new_x + new_width] = resized_image
            # 添加已经粘贴的图像信息到列表中
            pasted_images.append((new_x, new_y, new_width, new_height))

            # 解析对应的注释文件
            annotation_path = os.path.join(annotations_folder, os.path.splitext(image_file)[0] + ".xml")
            objects = parse_annotation(annotation_path)

            if objects:
                # 在注释中添加目标物体的信息
                for obj_name, xmin, ymin, xmax, ymax in objects:
                    obj = ET.SubElement(root, "object")
                    ET.SubElement(obj, "name").text = obj_name
                    ET.SubElement(obj, "pose").text = "Unspecified"
                    ET.SubElement(obj, "truncated").text = "0"
                    ET.SubElement(obj, "difficult").text = "0"
                    bndbox = ET.SubElement(obj, "bndbox")
                    # 根据图像的缩放和粘贴位置调整目标物体的边界框坐标
                    ET.SubElement(bndbox, "xmin").text = str(int(xmin * scaling_factor) + new_x)
                    ET.SubElement(bndbox, "ymin").text = str(int(ymin * scaling_factor) + new_y)
                    ET.SubElement(bndbox, "xmax").text = str(int(xmax * scaling_factor) + new_x)
                    ET.SubElement(bndbox, "ymax").text = str(int(ymax * scaling_factor) + new_y)

            # 从图像文件列表中移除已使用的图像文件
            image_files.remove(image_file)

        # 将合成图像保存到输出文件夹中
        cv2.imwrite(os.path.join(output_folder, f"output_{i+1}.jpg"), canvas)

        # 将注释文件保存到输出文件夹中
        xml_string = ET.tostring(root)
        dom = parseString(xml_string)
        pretty_xml_as_string = dom.toprettyxml(indent="\t")
        with open(os.path.join(output_folder, f"output_{i+1}.xml"), "w") as f:
            f.write(pretty_xml_as_string)

create_images("background.png",500)  # 创建50张图像

