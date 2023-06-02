import pathlib
import os
from ssg_utils.cv.dataset_utils import get_filenames_for_label

curr_file_path = pathlib.Path(__file__).parent 
annotations_path = os.path.join(curr_file_path,"data","IndianVehicleDataset","annotations")
label_name = "mini truck"

filenames = get_filenames_for_label(annotations_path=annotations_path, label_name=label_name)

print(filenames)