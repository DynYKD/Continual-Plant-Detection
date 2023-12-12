import json
import os
import pycocotools.mask as mask_utils
from PIL import Image, ImageDraw
import numpy as np
import re

def find_bbox_coordinates(shape):
    x_points = [x for (x,_) in shape["points"]]
    y_points = [y for (_,y) in shape["points"]]
    x_min, x_max = int(np.min(x_points)), int(np.max(x_points))
    y_min, y_max = int(np.min(y_points)), int(np.max(y_points))
    bbox = [x_min, y_min, x_max-x_min, y_max-y_min]
    return bbox

def split(string):
    match = re.search('\d', string)
    if match:
        index = match.start()
        return [string[:index], string[index:]]
    else:
        return [string]

def find_category_id(label, category_list):
    reformat_label = label.lower().replace(" ", "_")
    category_id = int(np.where(category_list == reformat_label)[0][0]+1)
    return category_id

def build_json_file(dataset_path, category_list):
    all_json = [f for f in os.listdir(dataset_path) if f.endswith(".json")]
    images = []
    annotations = []
    anno_id = 1
    image_id = 1
    for anno in all_json:
        json_anno = json.load(open(os.path.join(dataset_path, anno)))
        width = json_anno["imageWidth"]
        height = json_anno["imageHeight"]
        file_name = json_anno["imagePath"]
        new_image = {"id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": file_name}
        images.append(new_image)
        

        for shape in json_anno["shapes"]:
            points = [tuple((x, y)) for (x, y) in json_anno["shapes"][0]["points"]]
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            img = Image.fromarray(binary_mask)
            draw = ImageDraw.Draw(img)
            draw.polygon(points, fill=1)
            binary_mask = np.array(img)
            segmentation = mask_utils.encode(np.asfortranarray(binary_mask))
            segmentation["counts"] = segmentation["counts"].decode('ascii')

            area = float(binary_mask.sum())
            bbox = find_bbox_coordinates(shape)
            category_id = find_category_id(shape["label"], category_list)
            new_anno = {"id": anno_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": segmentation,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0}
            annotations.append(new_anno)
            anno_id +=1
        image_id += 1
    categories = [{"id": i+1} for i in range(len(category_list))]
    json_file = {"images": images,
                 "annotations": annotations,
                 "categories":categories}
    return json_file

if __name__ == "__main__":
    json_template = json.load(open("../../MMA/data/pascal_sbd_train.json"))
    json_template.keys()

    dataset_path = os.path.join("../strawberry_diseases/data", "train")
    category_list = np.unique([split(f)[0] for f in os.listdir(dataset_path)])

    splits = ["train", "val", "test"]
    for s in splits:
        dataset_path = os.path.join("../strawberry_diseases/data", s)
        json_file = build_json_file(dataset_path, category_list)
        with open('data/strawberry_diseases/{}_data.json'.format(s), 'w') as f:
            json.dump(json_file, f)

    [anno["bbox"] for anno in json_file["annotations"]]
    json_file["annotations"][0]["segmentation"]

    len(json_file["images"])
    len(json_file["annotations"])