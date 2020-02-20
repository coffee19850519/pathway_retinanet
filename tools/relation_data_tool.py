import os,copy,torch
import numpy as np
#from detectron2.structures import BoxMode, RotatedBoxes
import json
import pandas as pd
import cv2
from shutil import copyfile
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures import Instances, RotatedBoxes,BoxMode
from tools.label_file import LabelFile
from sklearn.model_selection import train_test_split
from detectron2.utils.visualizer import Visualizer

# write a function that loads the dataset into detectron2's standard format
def get_annotation_dicts(json_path, img_path, category_list):
    #go through all label files
    dataset_dicts = []

    for idx, json_file in enumerate(os.listdir(json_path)):
        if os.path.splitext(json_file)[1] != '.json':
            continue
        #load json content
        try:
            imgs_anns = LabelFile(os.path.join(json_path, json_file))
            #read key and value from current json file
            filename = os.path.join(img_path, imgs_anns.imagePath)
            img = cv2.imread(filename)
            height, width = img.shape[:2]
            del img
        except Exception as e:
            print(str(e))
            continue

        #declare a dict variant to save the content
        record = {}

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for anno in imgs_anns.shapes:
            #assert not anno["label"]
            #anno = anno["label"]
            poly_points = np.array(anno['points'],np.float32).reshape((-1 , 2))
            #rotated_rect = cv2.minAreaRect(poly_points)
            px = poly_points[:, 0]
            py = poly_points[:, 1]
            # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            # poly = list(itertools.chain.from_iterable(poly))
            try:
                component = list(anno['component'])
            except:
                component = []

            try:
                #only extract valid annotations
                category_id = imgs_anns.generate_category_id(anno,category_list)
            except Exception as e:
                print(str(e))
                continue

            obj = {
                #"bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                #"bbox_mode": BoxMode.XYXY_ABS,

                "bbox": anno['rotated_box'],
                "bbox_mode": -1,
                #"segmentation": [poly],
                "component": component,
                "category_id":  category_id,
                "iscrowd": 0

            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def split_data_into_train_and_validation_Kfold(json_path, validation_ratio = 0.2, K = 10):

    #image_list = os.listdir(image_path)
    json_list = os.listdir(json_path)
    #image_list.sort()
    #json_list.sort()
    #print(image_list)
    #print(json_list)
    for idx in range(K):
        #image_train, image_validation, \
        json_train, json_validation = train_test_split(json_list, test_size= validation_ratio)
        #generate train_k and val_k folder

        if not os.path.exists(os.path.join(json_path,r'train_'+ str(idx))):
            os.mkdir(os.path.join(json_path,r'train_'+ str(idx)))
        if not os.path.exists(os.path.join(json_path,r'val_'+ str(idx))):
            os.mkdir(os.path.join(json_path,r'val_'+ str(idx)))


        #copy train jsons files to different folders
        for json_file in json_train:
            copy_info = copyfile(os.path.join(json_path, json_file),
                     os.path.join(json_path, r'train_'+ str(idx), json_file))
            print('copied:' + copy_info + '\n')

        #copy val jsons files to different folders
        for json_file in json_validation:
            copy_info = copyfile(os.path.join(json_path, json_file),
                                 os.path.join(json_path, r'val_' + str(idx), json_file))
            print('copied:' + copy_info + '\n')

def register_pathway_dataset(json_path, img_path, category_list):
    for d in ["train", "val"]:
        DatasetCatalog.register("pathway_" + d,
                                lambda d=d: get_annotation_dicts(json_path + d, img_path,
                                                                 category_list))
        MetadataCatalog.get("pathway_" + d).set(thing_classes=category_list)

def register_Kfold_pathway_dataset(json_path, img_path, category_list, K = 10):
    for d in ["train", "val"]:
        for idx_fold in range(K):
            DatasetCatalog.register("pathway_" + d + '_' + str(idx_fold),
                                    lambda d=d: get_annotation_dicts(json_path + d + '_' + str(idx_fold), img_path,
                                                                     category_list))
            MetadataCatalog.get("pathway_" + d + '_' + str(idx_fold)).set(thing_classes=category_list)
            #MetadataCatalog.get("pathway_" + d + '_' + str(idx_fold)).set('coco')

class PathwayDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super(PathwayDatasetMapper, self).__init__(cfg, is_train)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            if not self.keypoint_on:
                anno.pop("keypoints", None)

        annos = [
                transform_rotated_boxes_annotations(obj, transforms)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
                ]

        instances = rotated_annotations_to_instances(annos, image_shape)

        # Create a tight bounding box from masks, useful when image is cropped
        if self.crop_gen and instances.has("gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        del annos, instances
        return dataset_dict



def rotated_annotations_to_instances(annos, image_size):
    boxes = [obj["bbox"] for obj in annos]
    boxes = torch.tensor(boxes, dtype=torch.float)
    target = Instances(image_size)
    boxes = target.gt_boxes = RotatedBoxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    # include component list into target
    # if len(annos) and "component" in annos[0]:
    #     component = []
    #     for obj in annos:
    #           torch.stack
    #         component.append(obj["component"])
    #     # component = np.array(component)
    #
    #     #component = torch.tensor(component, dtype=torch.int8)
    # target.gt_component = np.array(component)
    del boxes, classes
    return target

def transform_rotated_boxes_annotations(annotation, transforms):

    bbox = np.array(annotation["bbox"], np.float).reshape(-1, 5)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = transforms.apply_rotated_box(bbox)[0]
    annotation["bbox_mode"] = -1
    return annotation

    # def _include_relation_annotations(self, annotation):
    #     if not self.relation_on:
    #         annotation.pop('components')
    #         return annotation
    #
    #     # Handle relation annotations
    #     if not self._validate_components_in_relation(annotation):
    #         annotation.pop('components')
    #     return annotation
    #
    # def _validate_components_in_relation(self, annotation):
    #    return True


def visualize_rotated_prediction(img, metadata, predictions, shown_categories ,score_cutoff = 0):
    vis = Visualizer(img, metadata)

    # get targeted annotations to show
    boxes = []
    labels = []
    # get the specific categories to show
    for idx in range(len(predictions)):
        if  float(predictions.iloc[idx]["score"]) >= score_cutoff and predictions.iloc[idx]["category_id"] in shown_categories:
            boxes.append(predictions.iloc[idx]["bbox"])
            labels.append(predictions.iloc[idx]["category_id"])
    names = metadata.get("thing_classes", None)
    if names:
        labels = [names[i] for i in labels]
    boxes = np.array(boxes, np.float).reshape((-1, 5))
    vis_gt = vis.overlay_rotated_instances(labels=labels, boxes=boxes).get_image()
    del boxes, labels
    return vis_gt[:, :, ::-1]

def visualize_rotated_groundtruth(img, metadata, gts, shown_categories):
    vis = Visualizer(img, metadata)

    # get targeted annotations to show
    boxes = []
    labels = []
    # get the specific categories to show
    for gt in gts:
        if  gt["category_id"] in shown_categories:
            boxes.append(gt["bbox"])
            labels.append(gt["category_id"])
    names = metadata.get("thing_classes", None)
    if names:
        labels = [names[i] for i in labels]
    boxes = np.array(boxes, np.float).reshape((-1, 5))
    vis_gt = vis.overlay_rotated_instances(labels=labels, boxes=boxes).get_image()
    del boxes, labels
    return vis_gt[:, :, ::-1]


def visualize_relation_instances(coco_format_json_file, dataset_name, save_vis_path,shown_categories, cut_off):
    metadata = MetadataCatalog.get(dataset_name)
    datasetDict =  DatasetCatalog.get(dataset_name)
    coco_instances = json.load(open(coco_format_json_file, 'r'))
    #read all predictions regarding one input image
    predictions = pd.DataFrame(coco_instances)
    for sample_info in datasetDict:
        instances_on_sample = predictions.loc[predictions['image_id'] == sample_info['image_id']]
        img = cv2.imread(os.path.join(sample_info['file_name']))
        vis_img = visualize_rotated_prediction(img, metadata, instances_on_sample, shown_categories, cut_off)
        file_base_name = os.path.basename(sample_info['file_name'])
        cv2.imwrite(os.path.join(save_vis_path, file_base_name), vis_img)
        del vis_img,img
    del metadata,coco_instances

def visualize_rectangle_prediction(img, metadata, predictions, shown_categories ,score_cutoff = 0):
    vis = Visualizer(img, metadata)

    # get targeted annotations to show
    boxes = []
    labels = []
    # get the specific categories to show
    for idx in range(len(predictions)):
        if  float(predictions.iloc[idx]["score"]) >= score_cutoff and predictions.iloc[idx]["category_id"] in shown_categories:
            boxes.append(predictions.iloc[idx]["bbox"])
            labels.append(predictions.iloc[idx]["category_id"])
    names = metadata.get("thing_classes", None)
    if names:
        labels = [names[i] for i in labels]
    boxes = np.array(boxes, np.float).reshape((-1, 4))
    boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    vis_gt = vis.overlay_instances(labels=labels, boxes=boxes).get_image()
    del boxes, labels
    return vis_gt[:, :, ::-1]

def visualize_element_instances(coco_format_json_file, dataset_name, save_vis_path,shown_categories, cut_off):
    metadata = MetadataCatalog.get(dataset_name)
    datasetDict =  DatasetCatalog.get(dataset_name)
    element_instances = json.load(open(coco_format_json_file, 'r'))
    #read all predictions regarding one input image
    predictions = pd.DataFrame(element_instances)
    for sample_info in datasetDict:
        instances_on_sample = predictions.loc[predictions['image_id'] == sample_info['image_id']]
        img = cv2.imread(os.path.join(sample_info['file_name']))

        vis_img = visualize_rectangle_prediction(img, metadata, instances_on_sample, shown_categories, cut_off)
        file_base_name = os.path.basename(sample_info['file_name'])
        cv2.imwrite(os.path.join(save_vis_path, file_base_name), vis_img)
        del vis_img,img
    del metadata,element_instances

if __name__ == "__main__":

    #split all images
    # split_data_into_train_and_validation_Kfold(
    #         #r'/home/fei/Desktop/images/',
    #         r'/home/fei/Desktop/Henrys Annotations/',
    #         validation_ratio=0.25)

    # print(image_train)
    # print(json_train)

    # should be embedded into configer file
    #category_list = ['activate','gene','inihibit']
    category_list = ['activate_relation', 'inhibit_relation']
    img_path = r'/home/fei/Desktop/test/images/'
    json_path = r'/home/fei/Desktop/test/jsons/'


    # K = 10
    # for d in ["train", "val"]:
    #     #for idx_fold in range(K):
    #         idx_fold = 0
    #         DatasetCatalog.register("pathway_" + d + '_' + str(idx_fold), lambda d=d:get_annotation_dicts(json_path + d + '_' + str(idx_fold), img_path, category_list))
    #
    #         MetadataCatalog.get("pathway_" + d + '_' + str(idx_fold)).set(thing_classes=category_list)
    #split_data_into_train_and_validation_Kfold(json_path, validation_ratio=0.1, K=1)
    register_Kfold_pathway_dataset(json_path, img_path, category_list, K =1)
    # dicts = DatasetCatalog.get('pathway_val_0')
    # metadata = MetadataCatalog.get('pathway_val_0')
    # for dic in dicts:
    #     img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
    #     basename = os.path.basename(dic["file_name"])
    #     annotations = dic.get("annotations", None)
    #     vis_img = visualize_rotated_groundtruth(img, metadata, annotations,[3])
    #     #vis_gt = vis.draw_dataset_dict(dic).get_image()
    #     cv2.imwrite(os.path.join(r'/home/fei/Desktop/results/', basename), vis_img)
    #     del img, vis_img

    visualize_relation_instances(r'./output/relation/coco_instances_results_new.json',
                             'pathway_val_0',r'/home/fei/Desktop/vis_results/',[0,1],0.5)
