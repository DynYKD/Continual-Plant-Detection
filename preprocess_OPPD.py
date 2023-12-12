import os
import numpy as np
import json
from PIL import Image
import cv2
from segment_anything import build_sam, SamPredictor

os.listdir("OPPD/DATA")
category_list = [c.replace("\n","").split(":")[1] for c in open("OPPD/DATA/images_plants/labels.txt").readlines()]
#category_list.remove("SONAS")
#for c in classes:

#json_template = json.load(open("../MMA/data/pascal_sbd_train.json"))
#json_template.keys()

train_splits, val_splits, test_splits = [], [], []

for c in category_list:
    if c == "SONAS":
         train_splits.append([])
         val_splits.append([])
         test_splits.append([])
    else:
        dataset_path = "OPPD/DATA/images_full/{}".format(c)
        #all_annotations_path = "OPPD/DATA/images_plants/annotations_all.json"
        all_files = os.listdir(dataset_path)

        images_files = [f for f in all_files if f.endswith(".jpg")]
        x = [f.split("_")[1] for f in images_files]
        boxes = list(set(x))
        np.random.shuffle(boxes)
        n_per_split = len(boxes) // 3
        train, val, test = boxes[:n_per_split], boxes[n_per_split:2*n_per_split], boxes[2*n_per_split:]
        train_splits.append(train)
        val_splits.append(val)
        test_splits.append(test)
        print(len(train), len(val), len(test))


def build_json_file(splits, category_list, predictor=None):
    
    images = []
    annotations = []
    image_id = 1
    anno_id = 1
    #all_annotations = json.load(open(all_annotations_path))
    #splits = train_splits
    
    for s, c in enumerate(category_list):
        if c != "SONAS":
            dataset_path = "OPPD/DATA/images_full/{}".format(c)
            all_files = os.listdir(dataset_path)
            images_files = [f for f in all_files if f.endswith(".jpg")]
            images_split = [im_file for im_file in images_files if any([t in im_file for t in splits[s]])]

            for i in range(len(images_split)):
                file_name = images_split[i]
                im_name = file_name.split(".")[0]
                
                img = cv2.imread(os.path.join(dataset_path, file_name))
                width, height = 4096, 3000 #im.width, im.height
                new_image = {"id": image_id,
                            "width": width,
                            "height": height,
                            "file_name": os.path.join(c, file_name)}
                images.append(new_image)


                annotations_image = json.load(open(os.path.join(dataset_path, im_name+".json")))
                
                if predictor is not None:
                    boxes = [(anno["bndbox"]["xmin"], anno["bndbox"]["ymin"], anno["bndbox"]["xmax"], anno["bndbox"]["ymax"]) for anno in annotations_image["plants"]]
                    masks = segment_from_box(predictor, img, boxes)
                    import pdb
                    pdb.set_trace()
                    if masks is not []:
                        return masks
                for anno in annotations_image["plants"]:
                    category_id = category_list.index(c)+1
                    xmin, ymin, xmax, ymax = anno["bndbox"]["xmin"], anno["bndbox"]["ymin"], anno["bndbox"]["xmax"], anno["bndbox"]["ymax"]
                    bbox = [xmin, ymin, xmax-xmin, ymax-ymin]
                    new_anno = {"id": anno_id,
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": bbox,
                                "iscrowd": 0}
                    annotations.append(new_anno)
                    anno_id +=1

                image_id += 1 

    categories = [{"id": i+1} for i in range(len(category_list))]
    json_file = {"images": images,
             "annotations": annotations,
             "categories": categories}
    
    return json_file

def segment_from_box(predictor, img, boxes):
    predictor.set_image(img)
    all_masks = []
    for box in boxes:
        masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=False)
        best_mask = masks[np.argmax(scores)]
        all_masks.append(best_mask)
    return all_masks


splits = {"train": train_splits, "val": val_splits, "test": test_splits}
for s in ["train", "val", "test"]:
    json_file = build_json_file(splits[s], category_list)
    #with open('MMA/data/OPPD/{}_data.json'.format(s), 'w') as f:
    #        json.dump(json_file, f)