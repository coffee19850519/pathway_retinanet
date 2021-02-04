# -*- coding: utf-8 -*-
import itertools
import json
import os
import pandas as pd
import cv2
import argparse
import base64
from detectron2.data import DatasetFromList, MapDataset
from detectron2.structures import BoxMode,Boxes
from torchvision import datasets, transforms
import torch
from detectron2.config import get_cfg
from detectron2.data.build import DatasetMapper, trivial_batch_collator
from OCR import OCR
from fuzzywuzzy import fuzz, process
from fuzzywuzzy.process import default_processor
import cfg
import cfg_head
from detectron2.checkpoint import DetectionCheckpointer
from plain_train_net import ValidationSampler
from train_net import RegularTrainer,Trainer
from contextlib import contextmanager
import numpy as np
# from GCV import gcv_ocr
from tools.relation_data_tool import PathwayDatasetMapper
from tools.shape_tool import relation_covers_this_element
from formulate_relation import get_subimg, translation_transform_on_element_bbox, perspective_transform_on_element_bbox\
    ,find_largest_area_symbols,find_vertex_for_detected_relation_symbol_by_distance,dist_center,find_best_text,\
    center_point_in_box,calculate_distance_between_two_boxes
from swaps import swaps
from nfkc import nfkc
from deburr import deburr
from upper import upper
from expand import expand
# write a function that loads the dataset into detectron2's standard format
def get_data_dicts(img_path):
    # go through all label files
    dataset_dicts = []

    for idx, img_file in enumerate(os.listdir(img_path)):
        try:

            # read key and value from current json file
            filename = os.path.join(img_path, img_file)
            img = cv2.imread(filename)
            height, width = img.shape[:2]
            del img
        except Exception as e:
            # print(str(e))
            continue

        # declare a dict variant to save the content
        record = {}

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        record["annotations"] = None
        dataset_dicts.append(record)

    return dataset_dicts


def build_data_fold_loader(cfg, data_folder, mapper=None):
    """
    Similar to `build_detection_test_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),

    Args:
        cfg: a detectron2 CfgNode
        data_folder (str): folder includes data
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """

    dataset_dicts = list(itertools.chain.from_iterable([get_data_dicts(data_folder)]))
    # dataset_dicts = list(itertools.chain.from_iterable(get_data_dicts(data_folder)))
    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = ValidationSampler(len(dataset))

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, cfg.SOLVER.IMS_PER_BATCH, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    # del dataset_dicts, dataset
    return data_loader


def instances_to_coco_json(instances, img_id, file_name):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "file_name": file_name,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results


def inference_on_dataset(model, data_loader):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    # num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    # logger = logging.getLogger(__name__)
    # logger.info("Start inference on {} images".format(len(data_loader)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    predictions = []
    with inference_context(model.to(device)), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            # print('&&&&&',idx,inputs)
            output = model.to(device)(inputs)
            instances = output[0]["instances"].to(cpu_device)
            prediction = instances_to_coco_json(instances, inputs[0]["image_id"], inputs[0]['file_name'])
            predictions.extend(prediction)
            del prediction
            # print(prediction)
    return predictions


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


# def setup(cfg_file_path):
#     """
#     Create configs and perform basic setups.
#     """
#     cfg = get_cfg()
#     cfg.merge_from_file(cfg_file_path)
#     cfg.OUTPUT_DIR = os.path.join(r'./output/interface/')
#     # cfg.test_home_folder =kwargs['dataset']
#     cfg.freeze()
#
#     return cfg

def setup(cfg, kwargs):
    """
    Create configs and perform basic setups.
    """
    configuration = get_cfg()
    configuration.merge_from_file(cfg.element_config_file)
    # configuration.OUTPUT_DIR = os.path.join(r'./output/interface/')
    # configuration.test_home_folder = kwargs['dataset']
    # print('cfg.test_home_folder :',configuration.test_home_folder)
    configuration.freeze()

    return configuration


def predict(cfg_file_path,entity_type,data_folder):
    config =get_cfg()
    config.merge_from_file(cfg_file_path)
    config.freeze()

    if entity_type!= 'rotated_relation':
        model = RegularTrainer.build_model(config)
        DetectionCheckpointer(model=model,
                              save_dir=config.OUTPUT_DIR).resume_or_load(cfg.relation_model, resume=False)

        data_loader = build_data_fold_loader(config, data_folder, mapper=DatasetMapper(config, False))

        #img_size = (data_loader.dataset[0]['height'], data_loader.dataset[0]['width'])
        predictions = inference_on_dataset(model, data_loader)
        # evaluation_res = RegularTrainer.test(config, model,
        #              RegularEvaluator(config.DATASETS.TEST[0], config,
        #                  True, False, config.OUTPUT_DIR))
        #pass

    else:
        model = Trainer.build_model(config)
        DetectionCheckpointer(model=model,
                              save_dir=config.OUTPUT_DIR).resume_or_load(
            os.path.join(config.OUTPUT_DIR, cfg.rotated_relation_model), resume=False)

        data_loader = build_data_fold_loader(config, data_folder, mapper=PathwayDatasetMapper(config, False))
        #img_size = (data_loader.dataset[0]['height'], data_loader.dataset[0]['width'])
        predictions = inference_on_dataset(model, data_loader)

    del model,data_loader
    return predictions


def normalize_rect_vertex(points, image_size):
    if len(points) == 4:
        boxes = np.array(points, np.float).reshape((-1, 4))
        boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

        boxes = Boxes(boxes)
        boxes.clip(image_size)

        points = np.array(boxes.tensor).reshape((2, 2))
        # print('point',points)
        pt0 = np.min(points[:, 0])
        pt1 = np.min(points[:, 1])
        pt4 = np.max(points[:, 0])
        pt5 = np.max(points[:, 1])
        pt2 = pt4
        pt3 = pt1
        pt6 = pt0
        pt7 = pt5
        del points, boxes
        return np.array([[pt0, pt1], [pt2, pt3], [pt4, pt5], [pt6, pt7]], np.int32).reshape((4, 2))
    if len(points) == 5:
        cnt_x, cnt_y, w, h, angle = points
        return np.array(cv2.boxPoints(((cnt_x, cnt_y),(w, h), angle)), np.int32).reshape((4, 2))



def normalize_all_boxes(prediction_instances, image_size):
    assert 'normalized_bbox' in  prediction_instances.columns.values
    for row_idx in range(0, len(prediction_instances)):
        prediction_instances._set_value(prediction_instances.index[row_idx],
                                        'normalized_bbox',
                                        normalize_rect_vertex(prediction_instances.iloc[row_idx]['bbox'], image_size))
    #prediction_instances['normalized_bbox'] = prediction_instances['bbox'].map(normalize_rect_vertex, arg= (image_size))


def pair_gene(startor, startor_neighbor, receptor, receptor_neighbor, text_instances):

    assert receptor is not None
    assert startor is not None
    assert 'ocr' in text_instances.columns

    dist_ar = dist_center(startor, receptor)

    if startor_neighbor is None or \
            dist_center(startor_neighbor, startor) <= 0.1 * dist_ar:
        startor_neighbor = receptor


    if receptor_neighbor is None or \
            dist_center(receptor_neighbor, receptor) <= 0.1 * dist_ar:
        receptor_neighbor = startor

    best_startor_index = \
        find_best_text(startor, text_instances['perspective_bbox'], startor_neighbor, receptor)

    best_receptor_index = \
        find_best_text(receptor, text_instances['perspective_bbox'], receptor_neighbor, startor)

    if best_startor_index is not None and best_receptor_index is not None:
        dist_text = dist_center(center_point_in_box(text_instances.iloc[best_startor_index]['perspective_bbox']),
                                center_point_in_box(text_instances.iloc[best_receptor_index]['perspective_bbox']))

        if best_receptor_index != best_startor_index and dist_text > dist_ar * 0.8:
            return  text_instances.iloc[best_startor_index]['ocr'], \
                    text_instances.iloc[best_receptor_index]['ocr'], \
                    text_instances.iloc[best_startor_index]['perspective_bbox'], \
                    text_instances.iloc[best_receptor_index]['perspective_bbox']
        else:
            raise Exception('startor and receptor match to a same gene')
    else:
        raise Exception('cannot match startor or receptor')



# generate sub_image and fill entity bounding boxes for regular bbox
def generate_sub_image_bounding_relation_regular(img, relation_instance, element_instances_on_sample, offset):

    # image_name, image_ext = os.path.splitext(os.path.basename(img_file_name))
    # element_boxes= []
    # for idx in relation_instance['cover_entity']:
    #     element_boxes.append(entity_instances.iloc[idx]['normalized_bbox'])

    src_pts = relation_instance['normalized_bbox']

    # get all element instances on relation region
    try:
        element_instances_on_relation = element_instances_on_sample.iloc[relation_instance['covered_elements']].copy()

        # get bbox after perspective transform
        element_instances_on_relation['perspective_bbox'] = element_instances_on_relation['normalized_bbox'].apply(
            translation_transform_on_element_bbox, M=src_pts[0])

    except:
        print('element_instances_on_sample:', element_instances_on_sample)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped_img = get_subimg(img, src_pts, offset)


    return warped_img, element_instances_on_relation

# generate sub_image and fill entity bounding boxes
def generate_sub_image_bounding_relation_rotated(img, relation_instance, element_instances_on_sample, offset):

    # image_name, image_ext = os.path.splitext(os.path.basename(img_file_name))

    # element_boxes= []
    # for idx in relation_instance['cover_entity']:
    #     element_boxes.append(entity_instances.iloc[idx]['normalized_bbox'])

    src_pts = relation_instance['normalized_bbox']

    # corrdinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, relation_instance['bbox'][3]-offset],
                        [0, 0],
                        [relation_instance['bbox'][2]-offset, 0],
                        [relation_instance['bbox'][2]-offset,
                         relation_instance['bbox'][3]-offset]], dtype= np.float32)

    # the perspective transformation matrix
    transform = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts)
    # get all element instances on relation region
    element_instances_on_relation = element_instances_on_sample.iloc[relation_instance['covered_elements']].copy()

    # get bbox after perspective transform
    element_instances_on_relation['perspective_bbox'] = element_instances_on_relation['normalized_bbox'].apply(perspective_transform_on_element_bbox, M =transform)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped_img = cv2.warpPerspective(img, transform, (int(relation_instance['bbox'][2]),
                                                      int(relation_instance['bbox'][3])))

    return warped_img, element_instances_on_relation




def generate_relation_sub_image_and_pairing(img,image_name,image_ext,relation_instances_on_sample,element_instances_on_sample,relation_type,subimage_path):
    sub_image_path= os.path.join(subimage_path,'sub_image')
    paired_image_path=os.path.join(subimage_path,'paired')
    if not os.path.exists(sub_image_path):
        os.mkdir(sub_image_path)

    if not os.path.exists(paired_image_path):
        os.mkdir(paired_image_path)

    for relation_index in range(0, len(relation_instances_on_sample)):
        # plot covered elements by this relation at whole image
        element_instances_on_relation = element_instances_on_sample.iloc[relation_instances_on_sample.iloc[relation_index]['covered_elements']].copy()
        # print('element_instances_on_relation',element_instances_on_relation)
        covered_element_bboxes = element_instances_on_sample.iloc[relation_instances_on_sample.iloc[relation_index]['covered_elements']]['normalized_bbox']

        if relation_type =='relation':
            #regular bbox
            sub_img, element_instances_on_relation = generate_sub_image_bounding_relation_regular(img,
                                                 relation_instances_on_sample.iloc[relation_index],
                                                 element_instances_on_sample, offset= 1)
            # print('element_instances_on_relation',element_instances_on_relation)

        elif relation_type =='rotated_relation':
            # #rotated bbox
            sub_img, element_instances_on_relation = generate_sub_image_bounding_relation_rotated(img,
                                                  relation_instances_on_sample.iloc[relation_index],
                                                  element_instances_on_sample,  offset= 1)

        # plot elements on sub_img using their perspective

        sub_img_copy = sub_img.copy()
        # for element_idx in range(0, len(element_instances_on_relation)):
        #     if element_instances_on_relation.iloc[element_idx]['category_id'] == cfg.element_list.index('gene'):
        #         cv2.polylines(sub_img_copy, [element_instances_on_relation.iloc[element_idx]['perspective_bbox']], isClosed=True,
        #                       color=[255, 0, 0], thickness=2)
        #     if element_instances_on_relation.iloc[element_idx]['category_id'] == cfg.element_list.index('gene'):
        #         cv2.putText(sub_img_copy, element_instances_on_relation.iloc[element_idx]['ocr'],
        #                     tuple(element_instances_on_relation.iloc[element_idx]['perspective_bbox'][0]),
        #                     fontFace= cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale= 2, color =(0,255,0), thickness= 2)

        # # save sub-image to visualize sub-img
        cv2.imwrite(os.path.join(sub_image_path,image_name + str(relation_index) + image_ext), sub_img_copy)


        # do gene pairing
        startor, receptor = get_gene_pairs_on_relation_sub_image( sub_img, paired_image_path, 
            element_instances_on_relation= element_instances_on_relation, image_name=image_name, 
            image_ext=image_ext, idx=relation_index)
        # startor_index=np.where(element_instances_on_sample.normalized_bbox==(element_instances_on_sample.iloc[12]['normalized_bbox']))

        #update paired results into relation_instances_on_sample
        if startor is not None and receptor is not None:
            relation_instances_on_sample.at[relation_index, 'startor'] = startor
            relation_instances_on_sample.at[relation_index,'relation_category'] = cfg.relation_list[relation_instances_on_sample.iloc[relation_index]['category_id']]
            relation_instances_on_sample.at[relation_index, 'receptor'] = receptor

        del sub_img, element_instances_on_relation

    return relation_instances_on_sample



def assign_roles_to_elements(gene_instances_on_sub_image, relation_head_instance_on_sub_image):
    assert len(gene_instances_on_sub_image) == 2
    # print('gene_instances_on_sub_image',gene_instances_on_sub_image)
    # print('relation_head_instance_on_sub_image',relation_head_instance_on_sub_image)
    element_distance0 = calculate_distance_between_two_boxes(relation_head_instance_on_sub_image['perspective_bbox'], gene_instances_on_sub_image.iloc[0]['perspective_bbox'])
    element_distance1 = calculate_distance_between_two_boxes(relation_head_instance_on_sub_image['perspective_bbox'], gene_instances_on_sub_image.iloc[1]['perspective_bbox'])

    if element_distance0 > element_distance1:
        #return gene_instances_on_sub_image.iloc[0]['ocr'] + '<' + relation_head_instance_on_sub_image['category'] + '>' + gene_instances_on_sub_image.iloc[1]['ocr'], \
        #return gene_instances_on_sub_image.iloc[0]['perspective_bbox'], gene_instances_on_sub_image.iloc[1]['perspective_bbox']
        return gene_instances_on_sub_image.iloc[0]['ocr'], gene_instances_on_sub_image.iloc[1]['ocr'], \
                gene_instances_on_sub_image.iloc[0]['perspective_bbox'], gene_instances_on_sub_image.iloc[1]['perspective_bbox'],

    else:
        #return gene_instances_on_sub_image.iloc[1]['ocr'] + '<' + relation_head_instance_on_sub_image['category'] + '>' + gene_instances_on_sub_image.iloc[0]['ocr'], \
        #return gene_instances_on_sub_image.iloc[1]['perspective_bbox'], gene_instances_on_sub_image.iloc[0]['perspective_bbox']
        return gene_instances_on_sub_image.iloc[1]['ocr'], gene_instances_on_sub_image.iloc[0]['ocr'], \
               gene_instances_on_sub_image.iloc[1]['perspective_bbox'], gene_instances_on_sub_image.iloc[0][
                   'perspective_bbox'],


def assign_roles_to_elements_body(gene_instances_on_sub_image, relation_head_instance_on_sub_image):
    assert len(gene_instances_on_sub_image) == 2
    # print('gene_instances_on_sub_image',gene_instances_on_sub_image)
    # print('relation_head_instance_on_sub_image',relation_head_instance_on_sub_image)
    # element_distance0 = calculate_distance_between_two_boxes(relation_head_instance_on_sub_image['perspective_bbox'], gene_instances_on_sub_image.iloc[0]['perspective_bbox'])
    # element_distance1 = calculate_distance_between_two_boxes(relation_head_instance_on_sub_image['perspective_bbox'], gene_instances_on_sub_image.iloc[1]['perspective_bbox'])
    center0_x, center0_y = center_point_in_box(gene_instances_on_sub_image.iloc[0]['normalized_bbox'])
    center1_x, center1_y = center_point_in_box(gene_instances_on_sub_image.iloc[1]['normalized_bbox'])
    center2_x = relation_head_instance_on_sub_image['head'][0]
    center3_x = relation_head_instance_on_sub_image['tail'][0]
    center2_y = relation_head_instance_on_sub_image['head'][1]
    center3_y = relation_head_instance_on_sub_image['tail'][1]
    # print(center0_x, center0_y,center1_x, center1_y)
    element_distance_head0 = np.sqrt((center0_x - center2_x) ** 2 + (center0_y - center2_y) ** 2)
    element_distance_tail0 = np.sqrt((center0_x - center3_x) ** 2 + (center0_y - center3_y) ** 2)
    element_distance_head1 = np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
    element_distance_tail1 = np.sqrt((center1_x - center3_x) ** 2 + (center1_y - center3_y) ** 2)
    # print(element_distance_head0 ,element_distance_head1, element_distance_tail0, element_distance_tail1)

    if  element_distance_head0 > element_distance_tail0 and element_distance_head1 < element_distance_tail1 :
        #return gene_instances_on_sub_image.iloc[0]['ocr'] + '<' + relation_head_instance_on_sub_image['category'] + '>' + gene_instances_on_sub_image.iloc[1]['ocr'], \
        #return gene_instances_on_sub_image.iloc[0]['perspective_bbox'], gene_instances_on_sub_image.iloc[1]['perspective_bbox']
        return gene_instances_on_sub_image.iloc[0]['ocr'], gene_instances_on_sub_image.iloc[1]['ocr'], \
                gene_instances_on_sub_image.iloc[0]['normalized_bbox'], gene_instances_on_sub_image.iloc[1]['normalized_bbox'],

    else:
        #return gene_instances_on_sub_image.iloc[1]['ocr'] + '<' + relation_head_instance_on_sub_image['category'] + '>' + gene_instances_on_sub_image.iloc[0]['ocr'], \
        #return gene_instances_on_sub_image.iloc[1]['perspective_bbox'], gene_instances_on_sub_image.iloc[0]['perspective_bbox']
        return gene_instances_on_sub_image.iloc[1]['ocr'], gene_instances_on_sub_image.iloc[0]['ocr'], \
               gene_instances_on_sub_image.iloc[1]['normalized_bbox'], gene_instances_on_sub_image.iloc[0][
                   'normalized_bbox'],

# takes sub_image_filenames and predicted classes and extracts the relationship type and pairs
# returns entity pairs in list of tuples and list of strings (format: "relationship_type:starter|receptor")
def get_gene_pairs_on_relation_sub_image (sub_img,subimage_path, element_instances_on_relation,image_name, image_ext, idx):

    #analyze the element distribution first
    gene_instances_on_relation = element_instances_on_relation.loc[element_instances_on_relation['category_id'] == cfg.element_list.index('gene')]
    relation_symbol_instances_on_relation = element_instances_on_relation.loc[element_instances_on_relation['category_id'] != cfg.element_list.index('gene')]

    #pick the mostlikely relation symbols if more than 1 relation symbol
    relation_head_instance, relation_symbol_contour = \
    find_largest_area_symbols(sub_img, gene_instances_on_relation,relation_symbol_instances_on_relation)

    # print('relation_head_instance',relation_head_instance)
    # print('relation_symbol_contour',relation_symbol_contour)


    #if more than 2 genes
    if len(gene_instances_on_relation) > 2:
        # TODO alternative strategy 1: cluster then into 2 groups

        # ongoing strategy: from the relation symbol to find the closest 2 genes
        vertex_candidates = cv2.approxPolyDP(relation_symbol_contour, epsilon=5, closed=True)

        startor_point, startor_neighbor, receptor_point, receptor_neighbor = \
        find_vertex_for_detected_relation_symbol_by_distance(sub_img, vertex_candidates, relation_head_instance['normalized_bbox'])

        try:
            startor, receptor, startor_bbox, receptor_bbox = \
                pair_gene(startor_point, startor_neighbor, receptor_point, receptor_neighbor, gene_instances_on_relation)

            #just for visualization
            # cv2.polylines(sub_img, [startor_bbox], isClosed= True,  color=(0, 255, 0), thickness= 2)
            # cv2.polylines(sub_img, [receptor_bbox], isClosed= True, color=(0, 0, 255), thickness=2)
            # cv2.imwrite(os.path.join(subimage_path,image_name + '_' + str(idx) + image_ext), sub_img)

            return startor, receptor
        except Exception as e:
            print(str(e))
            return None, None

    # at last leave only 2 genes/groups and 1
    else:
        startor,receptor, startor_bbox, receptor_bbox = \
        assign_roles_to_elements_body(gene_instances_on_sub_image= gene_instances_on_relation,
                                 relation_head_instance_on_sub_image= relation_head_instance)
        # cv2.polylines(sub_img, [startor_bbox], isClosed=True, color=(0, 255, 0), thickness=2)
        # cv2.polylines(sub_img, [receptor_bbox], isClosed=True, color=(0, 0, 255), thickness=2)
        # cv2.imwrite(os.path.join(subimage_path  , image_name + '_' + str(idx) + image_ext), sub_img)
        return startor, receptor


def compute_iou(box1, box2, wh=True):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0]), int(box1[1])
        xmax1, ymax1 = int(box1[0]+box1[2]), int(box1[1]+box1[3])
        xmin2, ymin2 = int(box2[0]), int(box2[1])
        xmax2, ymax2 = int(box2[0]+box2[2]), int(box2[1]+box2[3])
        # xmin1, ymin1 = int(box1[0]-box1[2]/2.0), int(box1[1]-box1[3]/2.0)
        # xmax1, ymax1 = int(box1[0]+box1[2]/2.0), int(box1[1]+box1[3]/2.0)
        # xmin2, ymin2 = int(box2[0]-box2[2]/2.0), int(box2[1]-box2[3]/2.0)
        # xmax2, ymax2 = int(box2[0]+box2[2]/2.0), int(box2[1]+box2[3]/2.0)
        center = [(xmax1+xmin1)/2,(ymax1+ymin1)/2]

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
    ## 计算两个矩形框面积
    area1 = (xmax1-xmin1) * (ymax1-ymin1)
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1]))
    iou = inter_area / (area1+area2-inter_area+1e-6)
    # print('iou,center',iou,center)
    return iou,center

def compute_dis(point1,point2):
    dis  = np.sqrt((point2[0]-point1[0]) ** 2 + (point2[1]-point1[1]) ** 2)
    return dis    

def run_model(cfg, relation_h, **kwargs):
    with open(cfg.dictionary_path) as gene_name_list_fp:
        gene_name_list = json.load(gene_name_list_fp)

    configuration = setup(cfg, kwargs)

    model = RegularTrainer.build_model(configuration)

    DetectionCheckpointer(model=model,
                          save_dir=configuration.OUTPUT_DIR).resume_or_load(cfg.element_model, resume=False)

    data_folder = os.path.join(kwargs['dataset'], 'img/')
    ocr_sub_img_folder = os.path.join(data_folder, 'ocr_sub_img')
    if not os.path.isdir(ocr_sub_img_folder):
        os.mkdir(ocr_sub_img_folder)

    data_loader = build_data_fold_loader(configuration, data_folder, mapper=DatasetMapper(configuration, False))
  
    img_size = {}
    img_size['image_size'] = [data_loader.dataset[0]['height'], data_loader.dataset[0]['width']]
    # print('data_loader.dataset[0]',data_loader.dataset[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    predictions = []
    with inference_context(model.to(device)), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            # print('&&&&&',idx,inputs)
            output = model.to(device)(inputs)
            instances = output[0]["instances"].to(cpu_device)
            predictions = instances_to_coco_json(instances, inputs[0]["image_id"], inputs[0]['file_name'])
    # predictions = inference_on_dataset(model, data_loader)

            element_instances = pd.DataFrame(predictions)
            # print('element_instances',element_instances)
            element_instances['ocr'] = None
            element_instances['head'] = None
            element_instances['tail'] = None

            element_instances_on_samples = element_instances.loc[(element_instances['score'] >= cfg.element_threshold) \
                                                                 & (element_instances['category_id'] == cfg.element_list.index(
                'gene'))]

            relation_symbol_instances_on_samples = element_instances.loc[(element_instances['score'] >= cfg.element_threshold) \
                                                                         & (element_instances[
                                                                                'category_id'] != cfg.element_list.index(
                'gene'))]
            # print('relation_symbol_instances_on_samples', relation_symbol_instances_on_samples)
          
            # do OCR
            file_list = set(element_instances_on_samples['file_name'])
            for file_name in file_list:
                image_name, _ = os.path.splitext(os.path.basename(file_name))
                print('doing ocr to file {:s}'.format(file_name))
                # img_id = element_instances_on_samples.loc[element_instances['file_name'] == file_name]['image_id'].iloc[0]
                img_id = \
                element_instances_on_samples[element_instances_on_samples['file_name'] == file_name]['image_id'].values[0]

                element_instances_on_sample = element_instances_on_samples.loc[element_instances_on_samples['file_name'] == file_name]

                ocr_results, all_results_dict, corrected_results_dict, fuzz_ratios_dict, coordinates_list, element_idx_list = \
                 OCR(file_name, ocr_sub_img_folder, element_instances_on_sample, user_words=gene_name_list)
                # print('coordinates_list',coordinates_list)
                # ocr_results, coordinates_list = gcv_ocr(file_name)
                # postprocessing
                # nfkc->deburr->upper->expand->swap

                postprocessing_ocr_results = []
                for r in ocr_results:
                    r = nfkc(r)
                    r = ''.join(r)
                    r = deburr(r)
                    r = ''.join(r)
                    r = upper(r)
                    r = ''.join(r)
                    r = expand(r)
                    r = ''.join(r)
                    pp_r = swaps(r)
                    pp_r = ''.join(pp_r)

                    postprocessing_ocr_results.append(pp_r)

                # print("\nocr_results\n",ocr_results)
                # print('\npostprocessing_ocr_results\n',postprocessing_ocr_results)


                ocr_prediction_results = []
                for k in range(1, len(postprocessing_ocr_results)):
                    result = {
                        "image_id": img_id,
                        "file_name": file_name,
                        "category_id": 1,
                        "bbox": BoxMode.convert(np.array([coordinates_list[k][0], coordinates_list[k][1]]).reshape((-1, 4)),BoxMode.XYXY_ABS, BoxMode.XYXY_ABS).tolist()[0],
                        "score": float(1),
                        "ocr": postprocessing_ocr_results[k]
                    }
                    ocr_prediction_results.append(result)

                new_element_instances_on_samples = pd.DataFrame(ocr_prediction_results)
                # new_element_instances_on_samples['bbox'] = element_instances['bbox']
                # print('new_element_instances_on_samples',new_element_instances_on_samples['bbox'])
                for i in range(0,len(new_element_instances_on_samples)):
                    new_element_instances_on_samples['bbox'][i][2]= new_element_instances_on_samples['bbox'][i][2] -new_element_instances_on_samples['bbox'][i][0]
                    new_element_instances_on_samples['bbox'][i][3]= new_element_instances_on_samples['bbox'][i][3] -new_element_instances_on_samples['bbox'][i][1]

                relation_symbol_instances_on_samples = pd.concat([new_element_instances_on_samples, relation_symbol_instances_on_samples],
                                              ignore_index=True)
                # print('relation_symbol_instances_on_samples',relation_symbol_instances_on_samples)
                # plot ocr results on images
                # current_img = cv2.imread(file_name)
                # for result_idx in range(1, len(ocr_results)):
                #     cv2.putText(current_img, ocr_results[result_idx].encode('utf-8').decode('utf-8'),
                #                 tuple(np.array(coordinates_list[result_idx][0], np.int)),
                #                 cv2.FONT_HERSHEY_COMPLEX_SMALL,
                #                 1, (0, 0, 255), 1)
                #     # add ocr results to dataframe of element prediction results
                #     # set_ocr_results_to_element_instance_df(element_instances,
                #     #                                        element_instances_on_sample.iloc[element_idx_list[result_idx]]['bbox'],
                #     #                                        results[result_idx])
                # img_name, img_ext = os.path.splitext(file_name)
                # cv2.imwrite(img_name + '_ocr' + img_ext, current_img)
                # del current_img

                # save results
                json_dicts = []
                json_dicts.append(img_size)
                for i in range(1, len(postprocessing_ocr_results)):
                    json_dict = {}
                    json_dict['gene_name'] = postprocessing_ocr_results[i]
                    json_dict['coordinates'] = \
                    BoxMode.convert(np.array([coordinates_list[i][0], coordinates_list[i][1]]).reshape((-1, 4)),
                                    BoxMode.XYXY_ABS, BoxMode.XYXY_ABS).tolist()[0]
                    # add ocr results to dataframe of element prediction results
                    # set_ocr_results_to_element_instance_df(element_instances, element_instances_on_sample.iloc[element_idx_list[i]]['bbox'], results[i])
                    json_dicts.append(json_dict)
                # with open(data_folder + 'output-1.json', 'w+', encoding='utf-8') as file:
                #     json.dump(json_dicts, file, ensure_ascii=False)
                # with open(data_folder + 'output-1.json', 'w+', encoding='utf-8') as file:
                #     json.dump(json_dicts, file)
                with open(data_folder + '{:s}_elements.json'.format(image_name), 'w+', encoding='utf-8') as file:
                    json.dump(json_dicts, file)

                del ocr_results, coordinates_list, json_dicts,postprocessing_ocr_results

            element_instances = relation_symbol_instances_on_samples
            relation_subimage_path = os.path.join(data_folder, 'relation_subimage')
            if not os.path.isdir(relation_subimage_path):
                os.mkdir(relation_subimage_path)

            element_instances['normalized_bbox'] = None
            element_instances['center'] = None
            element_instances['startor'] = None
            element_instances['receptor'] = None
            element_instances["relation_category"] = None
            # todo: organize results from one image as a group and sort by their scores
            # element_instances.sort_values(by='score', ascending=False, inplace=True)

            # run relation prediction
            # relation_predictions = predict(cfg.relation_config_file, 'relation', data_folder)
            # # print('relation_predictions',relation_predictions[0])
            # # read the relations according to the same image
            # relation_instances = pd.DataFrame(relation_predictions)
            # print('relation_instances',relation_instances)
            # add startor and receptor columns into relation_instances

            # relation_instances['normalized_bbox'] = None
            # relation_instances['startor'] = None
            # relation_instances['relation_category'] = None
            # relation_instances['receptor'] = None

            image_file_list = set(element_instances['file_name'])
            for current_image_file in image_file_list:
                image_name, image_ext = os.path.splitext(os.path.basename(current_image_file))
                element_instances_on_sample = element_instances[(element_instances['file_name'] == current_image_file) &
                                                                (element_instances['score'] >= cfg.element_threshold)]
                # print('element_instances_on_sample\n',element_instances_on_sample)
                # relation_instances_on_sample = relation_instances[(relation_instances['file_name'] == current_image_file) &
                #                                                   (relation_instances['score'] >= cfg.relation_threshold)]
                #find gene center 
                gene_dic = element_instances[(element_instances['file_name'] == current_image_file)&(element_instances['category_id']==1)]
                gene_list = gene_dic['bbox'].tolist()
                for i in range(0,len(gene_list)):
                    center = [int(gene_list[i][0]+gene_list[i][2]/2),int(gene_list[i][1]+gene_list[i][3]/2)]
                    element_instances_on_sample['center'][gene_dic.index[i]] = center

                # find relation head
                relationhead = relation_h[(relation_h['file_name'] == current_image_file)]

                relation_head = relationhead['bbox'].tolist()
                element_relation = element_instances[(element_instances['file_name'] == current_image_file)&(element_instances['category_id']!=1)]
                relation_body = element_relation['bbox'].tolist()
                # print(relation_body)
                for i in range(0,len(relation_body)):
                    iou=0
                    temp_iou=0
                    for j in range(0,len(relation_head)):
                        # print('relation_head[i],relation_body[j]',relation_head[i],relation_body[j])
                        temp_iou,center = compute_iou(relation_head[j],relation_body[i],True)
                        if temp_iou>iou:
                            iou = temp_iou
                            element_instances_on_sample['head'][element_relation.index[i]] = center
                        # else:
                        #     element_instances_on_sample['head'][element_relation.index[i]] = [0,0]
                # print("iou",center)
                # print('compute after relation_symbol_instances_on_samples', relation_symbol_instances_on_samples)
                
                # find relation tail
                # print('current_image_file',current_image_file)
                img = cv2.imread(current_image_file)
                for i in range(0,len(relation_body)):
                    bbox = relation_body[i]
                    dis_max = 0
                    crop_img = img[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
                    if crop_img is not None:
                    # cv2.imwrite('/mnt/detectron2/pathway_retinanet_weiwei_65k/test/crop_ok.jpg', crop_img)
                        gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
                        corners = cv2.goodFeaturesToTrack(gray,20,0.06,10)
                        # print(type(corners))
                        
                        if type(corners) is not 'NoneType':
                            # 返回的结果是[[ 311., 250.]] 两层括号的数组。
                            # corners.tolist()
                            # try:
                            #     corners = np.int0(corners)
                            # except:
                            #     continue
                            corners = np.int0(corners)
                            # print('!!!!!!!!!!!!!!!!!',corners)
                            tail = []
                            for j in corners:
                                x,y = j.ravel()
                                cv2.circle(crop_img,(x,y),3,255,-1)
                                raw_x = x+bbox[0]
                                raw_y = y+bbox[1]
                                try:
                                    head_x = element_instances_on_sample['head'][element_relation.index[i]][0]
                                    head_y = element_instances_on_sample['head'][element_relation.index[i]][1]
                                except:
                                    continue
                                dis = np.sqrt((raw_x - head_x) ** 2 + (raw_y - head_y) ** 2)
                                if dis > dis_max:
                                    dis_max = dis
                                    tail = [raw_x,raw_y]

                            element_instances_on_sample['tail'][element_relation.index[i]] = tail 
                            del tail
                        else:
                            element_instances_on_sample['tail'][element_relation.index[i]] = [0,0]
                            # continue
                # print('element_instances_on_sample\n',element_instances_on_sample[['category_id','bbox','normalized_bbox','head','tail',\
                # 	'startor','receptor']])
                # print('relation_instances_on_sample',relation_instances_on_sample)
                img = cv2.imread(current_image_file)
                height, width, _ = img.shape
                # normalize_all_boxes(relation_instances_on_sample, (height, width))
                normalize_all_boxes(element_instances_on_sample, (height, width))



                #pairing
                # with open('{:s}_elements.json'.format(os.path.join(data_folder, image_name)), 'r') as output_fp:
                #         results.to_json(output_fp, orient='index')
                with open('{:s}_elements.json'.format(os.path.join(data_folder,'json', image_name)),'r') as load_f:
                    load_dict = json.load(load_f)
                    print(load_dict[1])
                gene_element = []
                gene_name = []
                for i in range(1,len(load_dict)):
                    center = [int(load_dict[i]['coordinates'][0]+load_dict[i]['coordinates'][2]/2),\
                    int(load_dict[i]['coordinates'][1]+load_dict[i]['coordinates'][3]/2)]
                    # element_instances_on_sample['center'][gene_dic.index[i]] = center
                    name = load_dict[i]['gene_name']
                    gene_element.append(center)
                    gene_name.append(name)
                

                # gene_e = element_instances_on_sample[element_instances['category_id']==1]
                # gene_element = gene_e.tolist()
                relation_e = element_instances_on_sample[element_instances['category_id']!=1]
                relation_head = relation_e['head'].tolist()
                relation_tail = relation_e['tail'].tolist()
                print(gene_element)
                print(relation_head)
                print(relation_tail)

                # for i in range(0,len(relation_head)):
                #     dis_head = 1000
                #     for j in range(0,len(gene_element)):
                #         if relation_head[i]!= None and relation_head[i]!=[] and gene_element[j]!=None and gene_element[j]!=[]:
                #             dis = compute_dis(relation_head[i],gene_element[j])
                #             if dis<dis_head:
                #                 dis_head = dis
                #                 min_j = j
                #                 ocr = gene_name[j]
                #     element_instances_on_sample['receptor'][relation_e.index[i]] = ocr

                for i in range(0,len(relation_tail)):
                    dis_tail = 1000
                    for j in range(0,len(gene_element)):
                        if relation_tail[i]!= None and relation_tail[i]!=[] and gene_element[j]!=None and gene_element[j]!=[]:
                            dis = compute_dis(relation_tail[i],gene_element[j])
                            if dis<dis_tail :
                                dis_tail = dis
                                min_j = j
                                ocr = gene_name[j]
                    element_instances_on_sample['startor'][relation_e.index[i]] = ocr

                for i in range(0,len(relation_head)):
                    dis_head = 1000
                    for j in range(0,len(gene_element)):
                        if relation_head[i]!= None and relation_head[i]!=[] and gene_element[j]!=None and gene_element[j]!=[]:
                            dis = compute_dis(relation_head[i],gene_element[j])
                            if dis<dis_head and gene_name[j]!=element_instances_on_sample['startor'][relation_e.index[i]]:
                                dis_head = dis
                                min_j = j
                                ocr = gene_name[j]
                    element_instances_on_sample['receptor'][relation_e.index[i]] = ocr
                for i in range(0,len(element_instances_on_sample)):
                    if element_instances_on_sample['category_id'][i]==0:
                        element_instances_on_sample['relation_category'][i] = 'activate_relation'
                    if element_instances_on_sample['category_id'][i]==2:
                        element_instances_on_sample['relation_category'][i] = 'inhibit_relation'
                    # element_instances_on_sample['head'][i]
                    # element_instances_on_sample['tail'][i]
                print('element_instances_on_sample\n',element_instances_on_sample[['relation_category','ocr','normalized_bbox','center','head','tail',\
                    'startor','receptor']])

                # for i in range(0,len(relation_element)):
                #     dis_tail = 1000
                #     for j in range(0,len(gene_element)):
                #         dis = compute_dis(relation_element[i]['tail'],gene_element[j]['center'])
                #         if dis<dis_tail:
                #             dis_tail = dis
                #             min_j = j
                #             ocr = gene_element[j]['ocr']
                #         element_instances_on_sample['startor'][relation_element.index[i]] = ocr

                        
                # for rows in element_instances_on_sample.iterrows():
                #     if rows['category_id'] != 1:
                #     	print(rows)
                #     # 
                # print(element_instances_on_sample[['category_id','bbox','normalized_bbox','head','tail',\
                # 	'startor','receptor']])
                # visualize normalized bboxes to confirm detection results
                img_copy = img.copy()
                for element_idx in range(0, len(element_instances_on_sample)):
                    if element_instances_on_sample.iloc[element_idx]['category_id'] == 0:
                        if element_instances_on_sample.iloc[element_idx]['score'] >= cfg.element_threshold:
                            cv2.polylines(img_copy, [element_instances_on_sample.iloc[element_idx]['normalized_bbox']],
                                          isClosed=True, color=(255, 0, 0), thickness=2)
                    elif element_instances_on_sample.iloc[element_idx]['category_id'] == 1:
                        if element_instances_on_sample.iloc[element_idx]['score'] >= cfg.element_threshold:
                            cv2.polylines(img_copy, [element_instances_on_sample.iloc[element_idx]['normalized_bbox']],
                                          isClosed=True, color=(0, 255, 0), thickness=2)
                    elif element_instances_on_sample.iloc[element_idx]['category_id'] == 2:
                        if element_instances_on_sample.iloc[element_idx]['score'] >= cfg.element_threshold:
                            cv2.polylines(img_copy, [element_instances_on_sample.iloc[element_idx]['normalized_bbox']],
                                          isClosed=True, color=(0, 0, 255), thickness=2)
                for i in range(0,len(element_instances_on_sample[element_instances_on_sample['category_id']!=1])):
                    head = element_instances_on_sample[element_instances_on_sample['category_id']!=1]['head']
                    element_head = head.tolist()
                    tail = element_instances_on_sample[element_instances_on_sample['category_id']!=1]['tail']
                    element_tail = tail.tolist()
                    # print(element_head,element_tail)
                    if element_head[i]!=None and element_head[i]!=[] and element_tail[i]!=None and element_tail[i]!=[]:
                        x_head = int(element_head[i][0])
                        y_head = int(element_head[i][1])
                        x_tail = int(element_tail[i][0])
                        y_tail = int(element_tail[i][1])
                        # print(x_head,y_head,x_tail,y_tail)
                        cv2.circle(img_copy, (x_head,y_head), 6, (128,0,128), -1)
                        cv2.circle(img_copy, (x_tail,y_tail), 6,(0,255,255), -1)
                   
                    # cv2.circle(img_copy, (element_instances_on_sample['head'][i][0],element_instances_on_sample['head'][i][1]), 6, (128,0,128), 0)
                    # cv2.circle(img_copy, (element_instances_on_sample['tail'][i][0],element_instances_on_sample['tail'][i][1]), 6,(255,255,0), 0)
                    # # img_copy(element_instances_on_sample['head'][i]) = [128,0,128] 
                    # img_copy(element_instances_on_sample['tail'][i]) = [255,255,0]
                # print('!!!!!!element_instances_on_sample',element_instances_on_sample)
                # for relation_idx in range(0, len(relation_instances_on_sample)):
                #     if relation_instances_on_sample.iloc[relation_idx]['category_id']==0:
                #         if relation_instances_on_sample.iloc[relation_idx]['score'] >= cfg.relation_threshold:
                #             cv2.polylines(img_copy, [relation_instances_on_sample.iloc[relation_idx]['normalized_bbox']],
                #                             isClosed=True, color=(255, 215, 0), thickness=2)
                #     elif relation_instances_on_sample.iloc[relation_idx]['category_id']==1:
                #         if relation_instances_on_sample.iloc[relation_idx]['score'] >= cfg.relation_threshold:
                #             cv2.polylines(img_copy, [relation_instances_on_sample.iloc[relation_idx]['normalized_bbox']],
                #                             isClosed=True, color=(128, 0, 128), thickness=2)

                # print('*******************************',relation_instances_on_sample)
                
                cv2.imwrite(os.path.join(relation_subimage_path, image_name + image_ext), img_copy)
                del img_copy
            result = element_instances_on_sample[element_instances_on_sample['category_id']!=1]
            results = result[["image_id","file_name","category_id","bbox","normalized_bbox","startor","relation_category","receptor"]]

            
            with open('{:s}_relation.json'.format(os.path.join(data_folder, image_name)), 'w') as output_fp:
                        results.to_json(output_fp, orient='index')
            del predictions, element_instances , image_file_list

def run_model_head(cfg_head,  **kwargs):

    configuration = setup(cfg_head, kwargs)

    model = RegularTrainer.build_model(configuration)

    DetectionCheckpointer(model=model,
                          save_dir=configuration.OUTPUT_DIR).resume_or_load(cfg_head.element_model, resume=False)

    # data_folder = r'/home/fei/Desktop/weiwei/data/use_case/test/0/'
    # ocr_sub_img_folder=r'/home/fei/Desktop/weiwei/data/use_case/test/ocr_sub_img/pdf_170_Targeting_2_10/'
    data_folder = os.path.join(kwargs['dataset'], 'img/')
    # print('data_folder:', data_folder)

    data_loader = build_data_fold_loader(configuration, data_folder, mapper=DatasetMapper(configuration, False))
    img_size = {}
    img_size['image_size'] = [data_loader.dataset[0]['height'], data_loader.dataset[0]['width']]

    predictions = inference_on_dataset(model, data_loader)

    element_instances = pd.DataFrame(predictions)
    # print('element_instances',element_instances)
    element_instances['ocr'] = None
    # element_instances_on_samples = element_instances.loc[(element_instances['score'] >= cfg.element_threshold) \
    #                                                      & (element_instances['category_id'] == cfg.element_list.index(
    #     'gene'))]
    # print(' element_instances_on_samples', element_instances_on_samples)
    relation_symbol_instances_on_samples = element_instances.loc[(element_instances['score'] >= cfg.element_threshold) \
                                                                 & (element_instances[
                                                                        'category_id'] != cfg.element_list.index(
        'gene'))]
    # print('head_relation_symbol_instances_on_samples', relation_symbol_instances_on_samples)



    return  relation_symbol_instances_on_samples



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='', help='input data')
    # parser.add_argument("--outputpath", type=str, default="", help="append to the dir name")
    # parser.add_argument("--cfg_file_path", type=str, default='', help="cfg_file_path")
    args = parser.parse_args()

    # args.outputpath = r'/home/fei/Desktop/test/results/'

    args.dataset = r'/mnt/detectron2/pathway_retinanet_weiwei_65k'
    # cfg_file_path = r'./Base-RetinaNet.yaml'

    # for k, v in vars(args).items():
    #     print(k, ':', v)

    # file_path='/home/fei/Desktop/weiwei/pathway_web/SkyEye/users/upload-files/2020-04-21T22:35:29.355Z/'
    file_path = vars(args)['dataset']
    img_path = os.path.join(file_path, 'img/')
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    # with open(file_path+'input.txt','r') as file:
    #     with open(img_path+'input.jpg','wb') as f:
    #         print('img_path:', img_path+'input.jpg')
    #
    #         img_data = file.read().split(',',1)[1]
    #         img=base64.b64decode(img_data)
    #         f.write(img)
    relation_head = run_model_head(cfg_head, **vars(args))
    run_model(cfg, relation_head, **vars(args))

