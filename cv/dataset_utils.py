import xml.etree.ElementTree as ET
import glob
import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

# Define aliases
FloatVector = list[float]
IntVector = list[int]


def xml_to_yolo_bbox(bbox: IntVector, w: int, h: int) -> FloatVector:
    """
    bbox: list of integers [xmin, ymin, xmax, ymax]
    w: Width of the input image
    h: Height of the input image 

    Returns list of floating point coordinates

    Ref: https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5
    """
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    """
    bbox: list of floats [x_center, y_center, width, height]
    w: Width of the input image
    h: Height of the input image 

    Returns list of integer point coordinates [xmin, ymin, xmax, ymax]

    Ref: https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5
    """
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


def draw_image(img, bboxes):
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=2)
    return img


class PascalVOCToYoloV5():

    """This class converts dataset in Pascal VOC XML format to Yolov5 format """

    def __init__(self, annotations_path: str, image_path: str, imgtype: str, class_path: str, output_path: str) -> None:
        """Define the attributes 
        class_path: path to store the class info
        """
        self.annotations_path = annotations_path
        self.image_path = image_path
        self.imgtype = imgtype
        self.class_path = class_path
        self.output_path = output_path
        self.classes = []
        self.annotations_to_delete = []
        # Create the output path and class if doesn't exist
        Path(class_path).mkdir(parents=True, exist_ok=True)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        # Check if the annotations and image path exist
        if os.path.exists(annotations_path) and os.path.exists(image_path):
            print("both annotations and image path exist")
        else:
            print("There should be two paths..annotations and image")

    def convert_and_store(self):

        # identify all the xml files in the annotations folder (input directory)
        files = glob.glob(os.path.join(self.annotations_path, '*.xml'))
        # loop through each
        for fil in files:
            basename = os.path.basename(fil)
            filename = os.path.splitext(basename)[0]
            print(f"Performing the Yolov5 conversion for {filename}.xml..")
            # check if the label contains the corresponding image file
            if not os.path.exists(os.path.join(self.image_path, f"{filename}.{self.imgtype}")):
                print(f"{filename} image does not exist!")
                continue

            result = []

            # parse the content of the xml file
            tree = ET.parse(fil)
            root = tree.getroot()
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)

            for obj in root.findall('object'):
                label = obj.find("name").text
                # check for new classes and append to list
                if label not in self.classes:
                    self.classes.append(label)
                index = self.classes.index(label)
                pil_bbox = [int(x.text) for x in obj.find("bndbox")]
                yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
                # convert data to string
                bbox_string = " ".join([str(x) for x in yolo_bbox])
                result.append(f"{index} {bbox_string}")

            if result:
                # generate a YOLO format text file for each xml file
                with open(os.path.join(self.output_path, f"{filename}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(result))
        print("Conversion is completed!..")
        with open(os.path.join(self.class_path, "Yolov5_classes.txt"), "w", encoding="utf-8") as f:
            for ind, element in enumerate(self.classes):
                f.write(f"{ind} {element}\n")


class VisualizeYoloV5:
    """This class helps to visualize the image object(s) with bounding box(es) given yolov5 label"""

    def __init__(self, image_path, label_path) -> None:
        self.image_path = image_path
        self.label_path = label_path
        self.bboxes = []

    def draw_bounding_boxes(self):
        img = Image.open(self.image_path)
        with open(self.label_path, 'r', encoding='utf8') as f:
            for line in f:
                data = line.strip().split(' ')
                bbox = [float(x) for x in data[1:]]
                self.bboxes.append(yolo_to_xml_bbox(
                    bbox, img.width, img.height))
        self.annot_img = draw_image(img, self.bboxes)

    def show_annotated_image(self):
        self.annot_img.show()

    def save_annotated_image(self):
        # get the image name
        img_name = self.image_path.split(os.sep)[-1]
        img_path = self.image_path.split(os.sep)[0]
        img_strp_name = img_name.split(".")[0]
        img_strp_ext = img_name.split(".")[-1]
        self.annot_img.save(os.path.join(
            img_path, img_strp_name+f"_annotated.{img_strp_ext}"))


class SplitYoloV5Dataset:
    """This class splits the dataset into train, val and test set"""

    def __init__(self, label_dir, image_dir, image_extension, split_save_path) -> None:
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.label_dir_name = "labels"+os.sep
        self.image_dir_name = "images"+os.sep
        self.lower_limit = 0
        self.imgext = image_extension
        self.split_save_path = split_save_path

    def copyfiles(self, fil, root_dir):
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]

        # copy image
        src = fil
        dest = os.path.join(root_dir, self.image_dir_name,
                            f"{filename}.{self.imgext}")
        shutil.copyfile(src, dest)

        # copy annotations
        src = os.path.join(self.label_dir, f"{filename}.txt")
        dest = os.path.join(root_dir, self.label_dir_name,
                            f"{filename}.txt")
        if os.path.exists(src):
            shutil.copyfile(src, dest)

    def perform_split(self):
        files = glob.glob(os.path.join(self.image_dir, f'*.{self.imgext}'))
        random.shuffle(files)

        folders = {"train": 0.8, "val": 0.1, "test": 0.1}
        check_sum = sum([folders[x] for x in folders])

        assert check_sum == 1.0, "Split proportion is not equal to 1.0"

        for folder in folders:
            try:
                # update folder path based on split_save_path
                folder_save = os.path.join(self.split_save_path, folder)
                Path(folder_save).mkdir(parents=True, exist_ok=False)
                temp_label_dir = os.path.join(folder_save, self.label_dir_name)
                Path(temp_label_dir).mkdir(parents=True, exist_ok=False)
                temp_image_dir = os.path.join(folder_save, self.image_dir_name)
                Path(temp_image_dir).mkdir(parents=True, exist_ok=False)
            except FileExistsError as e:
                print(
                    f"The folder '{folder}' already exists..please delete it for new split")
            limit = round(len(files) * folders[folder])
            for fil in files[self.lower_limit:self.lower_limit + limit]:
                self.copyfiles(fil, folder_save)
            self.lower_limit = self.lower_limit + limit


if __name__ == "__main__":

    curr_file_path = Path(__file__).parent.parent
    print(curr_file_path)
    annotations_path = os.path.join(
        curr_file_path, "data", "traffic_sign_detection", "annotations")
    print(annotations_path)
    image_path = os.path.join(curr_file_path, "data",
                              "traffic_sign_detection", "images")
    print(image_path)
    image_type = "png"
    class_path = os.path.join(curr_file_path, "data", "traffic_sign_detection")
    output_path = os.path.join(
        curr_file_path, "data", "traffic_sign_detection", "Yolov5")

    # create the object of the class
    pascal2yolov5 = PascalVOCToYoloV5(annotations_path=annotations_path, image_path=image_path,
                                      imgtype=image_type, class_path=class_path, output_path=output_path)
    # Perform the conversion
    pascal2yolov5.convert_and_store()

    # Test the visualization part
    img_name = "road64.png"
    full_img_path = os.path.join(image_path, img_name)
    full_label_path = os.path.join(output_path, img_name.split(".")[0]+".txt")
    visualize_obj = VisualizeYoloV5(
        image_path=full_img_path, label_path=full_label_path)
    visualize_obj.draw_bounding_boxes()
    visualize_obj.show_annotated_image()
    visualize_obj.save_annotated_image()

    split_save_path = os.path.join(curr_file_path, "data", "Yolov5_split")
    Path(split_save_path).mkdir(parents=True, exist_ok=True)
    # Test the dataset split
    split_obj = SplitYoloV5Dataset(
        label_dir=output_path, image_dir=image_path, image_extension=image_type, split_save_path=split_save_path)
    split_obj.perform_split()

    print("Dataset splitting completed!!")
