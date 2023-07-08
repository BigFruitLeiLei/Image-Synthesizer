import os
import cv2
import random
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from multiprocessing import Pool, Manager, Lock
from tqdm import tqdm

# ... Other definitions and functions remain the same ...
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





def create_single_image(args):
    filename, i, counter, lock, num_targets, max_attempts = args
    
    background = cv2.imread(filename)
    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    if mode == 0:
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    elif mode == 1:
            canvas = background_rgb.copy()
            canvas_height,canvas_width,channel = canvas.shape
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]

    root = ET.Element("annotation")
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
        image_file = random.choice(image_files)
        try:
            image = cv2.imread(os.path.join(input_folder, image_file))
            if image is None:
                continue
        except Exception as e:
            continue

        image_height, image_width = image.shape[:2]
        target_size = random.randint(min_target_size, max_target_size)
        scaling_factor = target_size / max(image_height, image_width)
        resized_image = cv2.resize(image, (int(image_width * scaling_factor), int(image_height * scaling_factor)))

        new_width, new_height = resized_image.shape[1], resized_image.shape[0]

        for attempt in range(max_attempts):
            new_x = random.randint(0, canvas_width - new_width)
            new_y = random.randint(0, canvas_height - new_height)

            if not is_overlapping(new_x, new_y, new_width, new_height, pasted_images):
                break
        else:
            print(f"Could not find a non-overlapping position for {image_file} after {max_attempts} attempts.")
            continue

        canvas[new_y:new_y + new_height, new_x:new_x + new_width] = resized_image
        pasted_images.append((new_x, new_y, new_width, new_height))

        annotation_path = os.path.join(annotations_folder, os.path.splitext(image_file)[0] + ".xml")
        objects = parse_annotation(annotation_path)

        if objects:
            for obj_name, xmin, ymin, xmax, ymax in objects:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = obj_name
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(xmin * scaling_factor) + new_x)
                ET.SubElement(bndbox, "ymin").text = str(int(ymin * scaling_factor) + new_y)
                ET.SubElement(bndbox, "xmax").text = str(int(xmax * scaling_factor) + new_x)
                ET.SubElement(bndbox, "ymax").text = str(int(ymax * scaling_factor) + new_y)

        image_files.remove(image_file)

    cv2.imwrite(os.path.join(output_folder, f"output_{i+1}.jpg"), canvas)

    xml_string = ET.tostring(root)
    dom = parseString(xml_string)
    pretty_xml_as_string = dom.toprettyxml(indent="\t")
    with open(os.path.join(output_folder, f"output_{i+1}.xml"), "w") as f:
        f.write(pretty_xml_as_string)
    with lock:
        counter.value += 1
    return

def create_images(filename, num_images, num_targets=15, max_attempts=50):
    with Manager() as manager:
        counter = manager.Value('i', 0)
        lock = manager.Lock()

        pbar = tqdm(total=num_images)  # Create a tqdm progress bar

        with Pool(processes=os.cpu_count()) as pool:
            result = pool.map_async(create_single_image, [(filename, i, counter, lock, num_targets, max_attempts) for i in range(num_images)])

            # While the pool is not done, update the tqdm progress bar according to the counter
            while not result.ready():
                pbar.n = counter.value
                pbar.refresh()

            # When the pool is done, set the tqdm progress bar to the total number of tasks
            pbar.n = pbar.total
            pbar.refresh()

        pbar.close()

if __name__ == "__main__":
    create_images(filename="background.png", num_images=500)
