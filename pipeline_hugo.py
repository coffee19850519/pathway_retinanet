# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import json
import cfg
import torch
from train_net import RegularTrainer
import pandas as pd
from nfkc import nfkc
from deburr import deburr
from upper import upper
from expand import expand
from swaps import swaps
import numpy as np
import copy
import requests
from math import floor, ceil
import nltk
import time
from xml.etree import ElementTree
import statistics
from corrected_ocr import corrected_processing_by_dict

from fuzzywuzzy import fuzz, process
from fuzzywuzzy.process import default_processor

from detectron2.structures import BoxMode,Boxes

# from GCV import gcv_ocr
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.build import DatasetMapper

# from nlp_pipeline.extract_text_from_pdf_then_xml_whole_article import convert_pdf_as_text_file, convert_txt_as_BioC_xml_file
# from nlp_pipeline.SubmitText_request import SubmitText_request
# from nlp_pipeline.SubmitText_retrieve import SubmitText
# from nlp_pipeline.get_gene_annotation_from_pubtator_result import extract_gene_annotation
# from nlp_pipeline_v2.from_PMCID_to_gene_annotation_and_cooccurrence.SubmitPMCIDList import SubmitPMCIDList
# from nlp_pipeline_v2.from_PMCID_to_gene_annotation_and_cooccurrence.gene_cooccurrence_from_pubtator_results import extract_gene_annotation_and_full_text,preprocess_sent_list_and_gene_list,gene_co_occurrence_in_sentence


from body_interface import instances_to_coco_json,setup,build_data_fold_loader,inference_context
from demo.predictor_jingyi import VisualizationDemo

from new_mmocr import mmocr_f, mmocr_without_det

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        # default="/home/19ljynenu/detectron2-master/pathway_retinanet_weiwei_65k/Base-RetinaNet.yaml",
        default="/root/lk/pathway_retinanet-pathway_retinanet_pairing/Base-RetinaNet.yaml",
        # default="/mnt/detectron2/pathway/Base-InstanceSegmentation.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        default="/root/lk/pathway_retinanet-pathway_retinanet_pairing/img/test.jpg",
        # nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        # default="/mnt/data/test/new.png",
        default="/root/lk/pathway_retinanet-pathway_retinanet_pairing/result/new.png",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        # default="/mnt/detectron2/pathway/output/model_0067600.pth",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def compute_dis(point1,point2):
    dis  = np.sqrt((point2[0]-point1[0]) ** 2 + (point2[1]-point1[1]) ** 2)
    return dis

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

def normalize_all_coordinates(prediction_instances, image_size):
    assert 'normalized_coordinates' in  prediction_instances.columns.values
    for row_idx in range(0, len(prediction_instances)):
        prediction_instances._set_value(prediction_instances.index[row_idx],
                                        'normalized_coordinates',
                                        normalize_rect_vertex(prediction_instances.iloc[row_idx]['coordinates'], image_size))

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

def reorganize_outputs(el_output,body_output,image_ids,file_names,cfg):
    '''

    Reorganize outputs into coco format and split results into dataframes by category id

    Args:
        el_output: direct outputs from the element model
        body_output: direct outputs from the relation model
        image_ids: image ids for current inputs
        file_names: filenames of current inputs
        cfg: configuration dict for model
    
    Return:
        (pd.DataFrame) element_instances: contains all detected element instances (ex. text)
        (pd.DataFrame) relation_head_instances: contains all detected arrow or t-bar heads
        (pd.DataFrame) relation_body_instances: contains all detected arrow or t-bar bodies

    '''

    # get results and convert to coco format
    cpu_device = torch.device("cpu")
    el_instances = el_output[0]["instances"].to(cpu_device)
    el_predictions = instances_to_coco_json(el_instances, image_ids, file_names)
    body_instances = body_output[0]["instances"].to(cpu_device)
    body_predictions = instances_to_coco_json(body_instances, image_ids, file_names)

    el_model_instances = pd.DataFrame(el_predictions)
    el_model_instances['ocr'] = None

    body_instances = pd.DataFrame(body_predictions)
    body_instances['ocr'] = None
    body_instances['head'] = None
    body_instances['tail'] = None


    # filter by score/confidence and predicted class
    element_instances = el_model_instances.loc[(el_model_instances['score'] >= cfg.element_threshold) \
                                                            & (el_model_instances['category_id'] == cfg.element_list.index('gene'))]
    relation_head_instances = el_model_instances.loc[(el_model_instances['score'] >= cfg.element_threshold) \
                                                            & (el_model_instances['category_id'] != cfg.element_list.index('gene'))]
    relation_body_instances = body_instances.loc[(body_instances['score'] >= cfg.relation_threshold) \
                                                            & (body_instances['category_id'] != cfg.element_list.index('gene'))]

    return element_instances, relation_head_instances, relation_body_instances

def get_jaccard(candidate_entity,gene_name_list,score_cutoff):

    candidate_entity = set(candidate_entity)

    results = []
    for gene_name in gene_name_list:
        jd = nltk.jaccard_distance(candidate_entity,set(gene_name))
        if jd > score_cutoff:
            results.append([gene_name,jd])

    return results

def jaro_distance(s1, s2):
     
    # If the s are equal
    if (s1 == s2):
        return 1.0
 
    # Length of two s
    len1 = len(s1)
    len2 = len(s2)
 
    # Maximum distance upto which matching
    # is allowed
    max_dist = floor(max(len1, len2) / 2) - 1
 
    # Count of matches
    match = 0
 
    # Hash for matches
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)
 
    # Traverse through the first
    for i in range(len1):
 
        # Check if there is any matches
        for j in range(max(0, i - max_dist),
                       min(len2, i + max_dist + 1)):
             
            # If there is a match
            if (s1[i] == s2[j] and hash_s2[j] == 0):
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break
 
    # If there is no match
    if (match == 0):
        return 0.0
 
    # Number of transpositions
    t = 0
    point = 0
 
    # Count number of occurrences
    # where two characters match but
    # there is a third matched character
    # in between the indices
    for i in range(len1):
        if (hash_s1[i]):
 
            # Find the next matched character
            # in second
            while (hash_s2[point] == 0):
                point += 1
 
            if (s1[i] != s2[point]):
                point += 1
                t += 1
    t = t//2
 
    # Return the Jaro Similarity
    return (match/ len1 + match / len2 +
            (match - t + 1) / match)/ 3.0

def get_jaro(candidate_entity,gene_name_list,score_cutoff):

    results = []
    for gene_name in gene_name_list:
        jd = jaro_distance(candidate_entity,gene_name)
        if jd > score_cutoff:
            results.append([gene_name,jd])

    return results


def get_ocr(current_image_file,article_gene_list,gene_name_list,data_folder,image_name,relation_body_instances,img_id,current_element_instances):

    '''

    Detect text from element predictions

    Args:
        current_image_file: filepath of image being processed
        article_gene_list: gene list from current image's article from pubtator
        gene_name_list: list of genes from general gene dictionary
        data_folder: folder to save results to
        image_name: name of image being processed
        relation_body_instances: arrow and t-bar body instances of current image
        img_id: image id of current image being processed
        
    
    Return:
        (pd.DataFrame) relation_body_instances: contains all detected element (ex. text) and arrow/t-bar body instances

    '''

    # postprocessing_ocr_results, coordinates_list = gcv_ocr(current_image_file)
    # postprocessing_ocr_results, coordinates_list = gcv_ocr(current_image_file,current_element_instances)

    # print(current_element_instances)

    # print('current_image_file:\n', current_image_file)

    # our model:
    box_res = {}
    # box_res['box'] = current_element_instances['bbox']#[round(x) for x in current_element_instances['bbox'][:]]  # round() 向上取整
    # print('box:\n', current_element_instances)
    current_element_instances = current_element_instances[current_element_instances['score'] > 0.9]
    box_res['box'] = current_element_instances['bbox']
    # print('box:\n', current_element_instances)
    ocr_results, coordinates_list = mmocr_without_det(img=current_image_file, element_bbox=box_res['box'], output='my_visualize_result/' +
                                                      current_image_file.split('/')[-1])
    ocr_results = ocr_results[0]


    # all mmocr:
    # ocr_results, coordinates_list = mmocr_f(img=current_image_file, output='mmocr_visualize_result/' +
    #                                                       current_image_file.split('/')[-1])
    # ocr_results = ocr_results[0]['text']
    # coordinates_list = coordinates_list[0]['box']


    postprocessing_ocr_results = []
    f = open('exHUGO_latest.json')
    hugo_dict = json.load(f)
    for idx, ocr_sample in enumerate(ocr_results):
        corrected_sample = corrected_processing_by_dict(hugo_dict,ocr_sample)
        # postprocessing_ocr_results[idx] = corrected_sample
        postprocessing_ocr_results.append(corrected_sample)
    # print('postprocessing_ocr_results:', postprocessing_ocr_results)



    # save results to json file
    # TODO:: this only saves the gene results
    current_img = cv2.imread(current_image_file)
    current_height, current_width, _ = current_img.shape
    json_dicts = []
    img_size = {}
    img_size['image_size'] = [current_height, current_width]
    json_dicts.append(img_size)
    for i in range(len(postprocessing_ocr_results)):
        json_dict = {}
        x1 = coordinates_list[i][0][0]
        y1 = coordinates_list[i][0][1]
        x2 = coordinates_list[i][1][0]
        y2 = coordinates_list[i][1][1]
        # json_dict['normalized_coordinates'] = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        json_dict['gene_name'] = ocr_results[i]
        json_dict['post_gene_name'] = postprocessing_ocr_results[i]
        json_dict['coordinates'] = \
            BoxMode.convert(np.array([coordinates_list[i][0], coordinates_list[i][1]]).reshape((-1, 4)),
                            BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).tolist()[0]

        # add ocr results to dataframe of element prediction results
        json_dicts.append(json_dict)

    gene_path = data_folder + 'gene_name/'

    if not os.path.exists(gene_path):
        os.makedirs(gene_path)
    with open(gene_path + '{:s}_elements.json'.format(image_name), 'w+', encoding='utf-8') as file:
        json.dump(json_dicts, file)

    # reorganize ocr result into coco json format for further use
    # TODO:: change coordinates to be true mins and maxes
    ocr_prediction_results = []
    for k in range(0, len(postprocessing_ocr_results)):
        x1 = coordinates_list[k][0][0]
        y1 = coordinates_list[k][0][1]
        x2 = coordinates_list[k][1][0]
        y2 = coordinates_list[k][1][1]
        result = {
            "image_id": img_id,
            "file_name": current_image_file,
            "category_id": 1,
            "normalized_bbox": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
            "bbox":
                BoxMode.convert(np.array([coordinates_list[k][0], coordinates_list[k][1]]).reshape((-1, 4)),
                                BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).tolist()[0],
            "score": float(1),
            "ocr": postprocessing_ocr_results[k]
        }
        ocr_prediction_results.append(result)

    # del ocr_results, coordinates_list, json_dicts, postprocessing_ocr_results
    del coordinates_list, json_dicts, postprocessing_ocr_results

    # combine ocr gene results and relation head results
    ocr_instances = pd.DataFrame(ocr_prediction_results)

    # uncomment if only want genes
    # remove zero index and subtract one, since ocr_prediction_results does not save first pred
    # not_gene_idxs.remove(0)
    # not_gene_idxs = [x-1 for x in not_gene_idxs]
    # ocr_instances = ocr_instances.drop(labels=not_gene_idxs,axis=0)

    # combine gene ocr and relation instances into 1 dataframe
    relation_body_instances = pd.concat([ocr_instances, relation_body_instances], ignore_index=True)

    # prepare df for building relationships
    relation_body_instances['normalized_bbox'] = None
    relation_body_instances['center'] = None
    relation_body_instances['startor'] = None
    relation_body_instances['receptor'] = None
    relation_body_instances['startor_bbox'] = None
    relation_body_instances['receptor_bbox'] = None
    relation_body_instances["relation_category"] = None

    return relation_body_instances

def get_relationship_head(processed_el_body_instances,current_relation_head_instances):

    '''

    For each arrow/t-bar body find corresponding head via largest IOU

    Args:
        processed_el_body_instances: element and arrow/t-bar body instances
        current_relation_head_instances: arrow/t-bar head instances
        
    Return:
        (pd.DataFrame) processed_el_body_instances: processed_el_body_instances with detected relationship heads

    '''

    # get relation head bbox
    relation_head_bboxes = current_relation_head_instances['bbox'].tolist()

    # get relation body bbox
    current_relation_body_instances = processed_el_body_instances[(processed_el_body_instances['category_id'] != 1)]
    relation_body_bboxes = current_relation_body_instances['bbox'].tolist()
    
    # head can be none for a relation body if no indicator is inside of it
    for i in range(0,len(relation_body_bboxes)):
        iou=0
        for j in range(0,len(relation_head_bboxes)):
            temp_iou,center = compute_iou(relation_head_bboxes[j],relation_body_bboxes[i],True)
            if temp_iou>iou:
                iou = temp_iou
                # relation_body_instance.index slice maintains the row indexing from processed_el_body_instance: this may throw a warning, but it is working as intended
                processed_el_body_instances['head'][current_relation_body_instances.index[i]] = center

    return processed_el_body_instances

def get_relationship_tail(current_image_file,processed_el_body_instances):

    '''

    Find each relation body's corresponding tail by choosing the detected corner furthest away from head in subimage as tail

    Args:
        current_image_file: filepath of image being processed
        processed_el_body_instances: element and arrow/t-bar body instances
        
    Return:
        (pd.DataFrame) processed_el_body_instances: processed_el_body_instances with detected relationship tails

    '''

    # get relation body bbox
    current_relation_body_instances = processed_el_body_instances[(processed_el_body_instances['category_id'] != 1)]
    relation_body_bboxes = current_relation_body_instances['bbox'].tolist()

    img = cv2.imread(current_image_file)
    for i in range(0, len(relation_body_bboxes)):
        bbox = relation_body_bboxes[i]
        dis_max = 0
        crop_img = img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        if crop_img is not None:

            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray, 20, 0.06, 10)

            if type(corners) is not 'NoneType':

                corners = np.int0(corners)

                tail = []
                for j in corners:
                    x, y = j.ravel()
                    cv2.circle(crop_img, (x, y), 3, 255, -1)
                    raw_x = x + bbox[0]
                    raw_y = y + bbox[1]

                    # TODO:: this try can be moved out, only needs to be run once every i loop
                    try:
                        head_x = processed_el_body_instances['head'][current_relation_body_instances.index[i]][
                            0]
                        head_y = processed_el_body_instances['head'][current_relation_body_instances.index[i]][
                            1]
                    except:
                        continue
                    dis = np.sqrt((raw_x - head_x) ** 2 + (raw_y - head_y) ** 2)
                    if dis > dis_max:
                        dis_max = dis
                        tail = [raw_x, raw_y]

                # current_relation_body_instances.index slice maintains the row indexing from processed_el_body_instances: this may throw a warning, but it is working as intended
                processed_el_body_instances['tail'][current_relation_body_instances.index[i]] = tail
                del tail
            else:
                # TODO:: handle this case better
                # if no corners found, set tail to top left corner
                processed_el_body_instances['tail'][current_relation_body_instances.index[i]] = [0, 0]

    return processed_el_body_instances

def get_receptor(relation_heads,current_relation_body_instances,gene_centers,processed_el_body_instances,processed_genes):

    '''

    Find each relation body's corresponding receptor by finding the closest gene

    Args:
        relation_heads: relationship head centers
        current_relation_body_instances: arrow/t-bar body instances
        gene_centers: center points of detected genes
        processed_el_body_instances: element and arrow/t-bar body instances
        processed_genes: element instances
        
    Return:
        (pd.DataFrame) processed_el_body_instances: processed_el_body_instances with detected gene receptors for each relationship
        (List) min_js: indices of detected receptors for each relationship

    '''

    min_js = []

    # print(len(relation_heads))

    # get relation body receptor
    for i in range(0,len(relation_heads)):

        dis_head = 500
        min_j = None
        min_dash = None
        min_dash_dist = 500
        min_entity = None
        min_entity_dist = 500
        for j in range(0,len(gene_centers)):
            if gene_centers[j]!=None and gene_centers[j]!=[]:
                dis = compute_dis(relation_heads[i],gene_centers[j])
                if dis<dis_head:
                    if processed_el_body_instances['ocr'][processed_genes.index[j]] == '-':
                        if dis < min_dash_dist:
                            min_dash_dist = dis
                            min_dash = j
                    else:
                        if dis < min_entity_dist:
                            min_entity_dist = dis
                            min_entity = j
        
        if min_entity is not None and min_dash is not None:
            entity_dis = compute_dis(relation_heads[i],gene_centers[min_entity])
            dash_dis = compute_dis(relation_heads[i],gene_centers[min_dash])
            dash_entity_dis = compute_dis(gene_centers[min_entity],gene_centers[min_dash])

            # do this compare to check if they are apart of same cluster
            if dash_entity_dis > 75:
                if entity_dis > dash_dis:
                    min_j = min_dash
                else:
                    min_j = min_entity
            else:
                if min_entity is not None:
                    min_j = min_entity
                else:
                    min_j = min_dash

        else:
            if min_entity is not None:
                min_j = min_entity
            else:
                min_j = min_dash

        min_js.append(min_j)
        try:
            ocr = processed_el_body_instances['ocr'][processed_genes.index[min_j]]
            processed_el_body_instances['receptor'][current_relation_body_instances.index[i]] = ocr
            processed_el_body_instances['receptor_bbox'][current_relation_body_instances.index[i]] = processed_el_body_instances['normalized_bbox'][processed_genes.index[min_j]]

        except:
            print(min_j)
            print(processed_genes.index[min_j])

    return processed_el_body_instances, min_js

def get_startor(relation_tails,current_relation_body_instances,gene_centers,processed_el_body_instances,processed_genes,min_js):

    '''

    Find each relation body's corresponding startor by finding the closest gene (no direct repeats)

    Args:
        relation_tails: relationship tails centers
        current_relation_body_instances: arrow/t-bar body instances
        gene_centers: center points of detected genes
        processed_el_body_instances: element and arrow/t-bar body instances
        processed_genes: element instances
        min_js: indices of detected receptors for each relationship
        
    Return:
        (pd.DataFrame) processed_el_body_instances: processed_el_body_instances with detected gene receptors for each relationship
        
    '''

    # print(len(relation_tails))

    # get relation body starter
    for i in range(0,len(relation_tails)):

        min_j = min_js[i]

        closes_entity = None
        dis_tail = 500
        min_dash = None
        min_dash_dist = 500
        min_entity = None
        min_entity_dist = 500
        for j in range(0,len(gene_centers)):

            # startor can't be the receptor
            if j == min_j:
                continue

            if gene_centers[j]!=None and gene_centers[j]!=[]:
                dis = compute_dis(relation_tails[i],gene_centers[j])
                if dis<dis_tail:
                    if processed_el_body_instances['ocr'][processed_genes.index[j]] == '-':
                        if dis < min_dash_dist:
                            min_dash_dist = dis
                            min_dash = j
                    else:
                        if dis < min_entity_dist:
                            min_entity_dist = dis
                            min_entity = j

        if min_entity is not None and min_dash is not None:
            entity_dis = compute_dis(relation_tails[i],gene_centers[min_entity])
            dash_dis = compute_dis(relation_tails[i],gene_centers[min_dash])
            dash_entity_dis = compute_dis(gene_centers[min_entity],gene_centers[min_dash])

            # do this compare to check if they are apart of same cluster
            if dash_entity_dis > 75:
                if entity_dis > dash_dis:
                    closes_entity = min_dash
                else:
                    closes_entity = min_entity

            else:
                if min_entity is not None:
                    closes_entity = min_entity
                else:
                    closes_entity = min_dash

        else:

            if min_entity is not None:
                closes_entity = min_entity
            else:
                closes_entity = min_dash

        try:

            tail_to_receptor = compute_dis(gene_centers[min_j],relation_tails[i])
            tail_to_closest_non_receptor = compute_dis(gene_centers[closes_entity],relation_tails[i])

            if tail_to_receptor < tail_to_closest_non_receptor:
                ocr = processed_el_body_instances['ocr'][processed_genes.index[closes_entity]]
                processed_el_body_instances['receptor'][current_relation_body_instances.index[i]] = ocr
                processed_el_body_instances['receptor_bbox'][current_relation_body_instances.index[i]] = processed_el_body_instances['normalized_bbox'][processed_genes.index[closes_entity]]

                ocr = processed_el_body_instances['ocr'][processed_genes.index[min_j]]
                processed_el_body_instances['startor'][current_relation_body_instances.index[i]] = ocr
                processed_el_body_instances['startor_bbox'][current_relation_body_instances.index[i]] = processed_el_body_instances['normalized_bbox'][processed_genes.index[min_j]]
            else:
                ocr = processed_el_body_instances['ocr'][processed_genes.index[closes_entity]]
                processed_el_body_instances['startor'][current_relation_body_instances.index[i]] = ocr
                processed_el_body_instances['startor_bbox'][current_relation_body_instances.index[i]] = processed_el_body_instances['normalized_bbox'][processed_genes.index[closes_entity]]


        except:
            print(closes_entity)
            print(min_dash)
            print(min_entity)
            print(processed_genes.index[closes_entity])
            print(processed_el_body_instances['receptor'][current_relation_body_instances.index[i]])

    return processed_el_body_instances

def save_visual(current_image_file, processed_el_body_instance,visuals_folder, image_name, ext):

    img = cv2.imread(current_image_file)
    height, width, _ = img.shape
    image_size = (height,width)

    # visualize normalized bboxes to confirm detection results and save
    img_copy = img.copy()
    for element_idx in range(0, len(processed_el_body_instance)):
        if processed_el_body_instance.iloc[element_idx]['category_id'] == 0:
            if processed_el_body_instance.iloc[element_idx]['score'] >= cfg.element_threshold:
                cv2.polylines(img_copy, [normalize_rect_vertex(processed_el_body_instance.iloc[element_idx]['bbox'], image_size)],
                                isClosed=True, color=(255, 0, 0), thickness=2)
        elif processed_el_body_instance.iloc[element_idx]['category_id'] == 1:
            if processed_el_body_instance.iloc[element_idx]['score'] >= cfg.element_threshold:
                cv2.polylines(img_copy, [normalize_rect_vertex(processed_el_body_instance.iloc[element_idx]['bbox'], image_size)],
                                isClosed=True, color=(0, 255, 0), thickness=2)
        elif processed_el_body_instance.iloc[element_idx]['category_id'] == 2:
            if processed_el_body_instance.iloc[element_idx]['score'] >= cfg.element_threshold:
                cv2.polylines(img_copy, [normalize_rect_vertex(processed_el_body_instance.iloc[element_idx]['bbox'], image_size)],
                                isClosed=True, color=(0, 0, 255), thickness=2)
    for i in range(0, len(processed_el_body_instance[processed_el_body_instance['category_id'] != 1])):
        head = processed_el_body_instance[processed_el_body_instance['category_id'] != 1]['head']
        element_head = head.tolist()
        tail = processed_el_body_instance[processed_el_body_instance['category_id'] != 1]['tail']
        element_tail = tail.tolist()

        if element_head[i] != None and element_head[i] != [] and element_tail[i] != None and element_tail[
            i] != []:
            x_head = int(element_head[i][0])
            y_head = int(element_head[i][1])
            x_tail = int(element_tail[i][0])
            y_tail = int(element_tail[i][1])

            cv2.circle(img_copy, (x_head, y_head), 6, (128, 0, 128), -1)
            cv2.circle(img_copy, (x_tail, y_tail), 6, (0, 255, 255), -1)

    cv2.imwrite(os.path.join(visuals_folder, image_name + ext), img_copy)
    del img_copy

def get_startors_and_receptors(current_image_file,processed_el_body_instances):

    '''

    Find each relation body's corresponding receptor and starter

    Args:
        current_image_file: filepath of image being processed
        processed_el_body_instances: element and arrow/t-bar body instances
        
    Return:
        (pd.DataFrame) processed_el_body_instances: processed_el_body_instances with detected gene starters and receptors for each relationship

    '''

    # normalize coords for pairing
    img = cv2.imread(current_image_file)
    height, width, _ = img.shape
    normalize_all_boxes(processed_el_body_instances, (height, width))

    # get current genes' center bboxes
    gene_dic = processed_el_body_instances[(processed_el_body_instances['category_id'] == 1)]
    gene_list = gene_dic['bbox'].tolist()
    for i in range(0, len(gene_list)):
        # gene_list bbox values are XYWH
        center = [int(gene_list[i][0] + gene_list[i][2] / 2), int(gene_list[i][1] + gene_list[i][3] / 2)]
        # gene_dic.index slice maintains the row indexing from element_instances: this may throw a warning, but it is working as intended
        processed_el_body_instances['center'][gene_dic.index[i]] = center
    processed_genes = processed_el_body_instances[processed_el_body_instances['category_id'] == 1]
    gene_centers = processed_genes['center'].tolist()

    # get relation body instances and their head & tail bboxes
    current_relation_body_instances = processed_el_body_instances[processed_el_body_instances['category_id'] != 1]
    relation_heads = current_relation_body_instances['head'].tolist()
    relation_tails = current_relation_body_instances['tail'].tolist()

    # get startors and receptors
    # TODO:: for given pair, double check startor and receptor selection minimizes total combo distancez
    processed_el_body_instances, min_js = get_receptor(relation_heads,current_relation_body_instances,gene_centers,processed_el_body_instances,processed_genes)
    processed_el_body_instances = get_startor(relation_tails,current_relation_body_instances,gene_centers,processed_el_body_instances,processed_genes,min_js)

    relation_list = []
    for index in processed_el_body_instances.index:
        if processed_el_body_instances.loc[index]['category_id'] == 0:
            # processed_el_body_instances.loc[index]['relation_category'] = 'activate_relation'
            relation_list.append('activate_relation')
        if processed_el_body_instances.loc[index]['category_id'] == 2:
            relation_list.append('inhibit_relation')
            # processed_el_body_instances.loc[index]['relation_category'] = 'inhibit_relation'
        if processed_el_body_instances.loc[index]['category_id'] == 1:
            relation_list.append('None')
    processed_el_body_instances['relation_category'] = relation_list
    return processed_el_body_instances

def score_by_cooccurrence(gene_co_occurrence,processed_el_body_instances):

    '''

    rank each relationship by how often their gene's co_occur the figure's article

    Args:
        gene_co_occurrence: current article's co-occurring genes by sentence
        processed_el_body_instances: element and arrow/t-bar body instances
        
    Return:
        (pd.DataFrame) processed_el_body_instances: processed_el_body_instances with detected gene starters and receptors for each relationship

    '''

    # TODO:: make this faster
    # set coocurrence score and relation category
    processed_el_body_instances['rank'] = 0
    r_body_instances = processed_el_body_instances[processed_el_body_instances['category_id'] != 1]
    for i, r_body_instance in r_body_instances.iterrows():

        for j, row in gene_co_occurrence.iterrows():
            if r_body_instance['startor'] == row['gene_name_1'] and r_body_instance['receptor'] == row['gene_name_2']:
                processed_el_body_instances['rank'][i] = row['co_occurrence']
                break
            elif r_body_instance['receptor'] == row['gene_name_1'] and r_body_instance['startor'] == row['gene_name_2']:
                processed_el_body_instances['rank'][i] = row['co_occurrence']
                break

        if r_body_instance['category_id']==0:
            processed_el_body_instances['relation_category'][i] = 'activate_relation'
        if r_body_instance['category_id']==2:
            processed_el_body_instances['relation_category'][i] = 'inhibit_relation'

    return processed_el_body_instances

def filter_heads_and_tails(processed_el_body_instances):

    '''

    remove relationships that have no head or tail detected

    Args:
        processed_el_body_instances: element and arrow/t-bar body instances
        
    Return:
        (pd.DataFrame) processed_el_body_instances: processed_el_body_instances with bad relationships removed

    '''

    current_relation_body_instances = processed_el_body_instances[processed_el_body_instances['category_id'] != 1]
    relation_heads = current_relation_body_instances['head'].tolist()
    relation_tails = current_relation_body_instances['tail'].tolist()
    rows_to_remove = []

    # if no head or tail, then don't want to save this relationship
    for i in range(0,len(relation_heads)):
        if relation_heads[i] == None or relation_heads[i] == []:
            rows_to_remove.append(current_relation_body_instances.index[i])
        elif relation_tails[i] == None or relation_tails[i] == []:
            rows_to_remove.append(current_relation_body_instances.index[i])

    processed_el_body_instances = processed_el_body_instances.drop(labels=rows_to_remove,axis=0)

    return processed_el_body_instances


def run_model(cfg, article_pd, **kwargs):

    article_pd = None
    
    # get gene dictionary
    with open(cfg.dictionary_path) as gene_name_list_fp:
        gene_name_list = json.load(gene_name_list_fp)
    gene_name_list = [x.upper() for x in gene_name_list]


    # load models
    configuration = setup(cfg, kwargs)

    body_model = RegularTrainer.build_model(configuration)
    DetectionCheckpointer(model=body_model, save_dir=configuration.OUTPUT_DIR).resume_or_load(cfg.relation_model, resume=False)

    el_model = RegularTrainer.build_model(configuration)
    DetectionCheckpointer(model=el_model, save_dir=configuration.OUTPUT_DIR).resume_or_load(cfg.element_model, resume=False)


    # create folders for post-processing
    data_folder = os.path.join(kwargs['dataset'], 'img/')
    # print('data_folder:', data_folder)
    # ocr_sub_img_folder = os.path.join(data_folder, 'ocr_sub_img')
    # if not os.path.isdir(ocr_sub_img_folder):
    #     os.mkdir(ocr_sub_img_folder)

    visuals_folder = os.path.join(data_folder, 'relation_subimage')
    if not os.path.isdir(visuals_folder):
        os.mkdir(visuals_folder)


    # get data loader
    data_loader = build_data_fold_loader(configuration, data_folder, mapper=DatasetMapper(configuration, False))

    # set which device to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set both models into eval mode and loop through data
    with inference_context(el_model.to(device)), inference_context(body_model.to(device)), torch.no_grad():
        for idx, inputs in enumerate(data_loader):

            # run inference
            el_output = el_model.to(device)(inputs)
            body_output = body_model.to(device)(inputs)

            # reorganize model outputs into dataframes by category
            image_ids = inputs[0]["image_id"]
            file_names = inputs[0]['file_name']
            element_instances, relation_head_instances, relation_body_instances = reorganize_outputs(el_output,body_output,image_ids,file_names,cfg)

            # print('element_instances:', element_instances)

            # loop through each image in batch
            file_list = set(element_instances['file_name'])
            for current_image_file in file_list:
                image_name, ext = os.path.splitext(os.path.basename(current_image_file))
                print('doing ocr to file {:s}'.format(current_image_file))

                # get current image's model outputs
                current_element_instances = element_instances[(element_instances['file_name'] == current_image_file)]
                # print(current_element_instances['bbox'][0])
                # print('current_element_instances:', current_element_instances)

                current_relation_head_instances = relation_head_instances[(relation_head_instances['file_name'] == current_image_file)]
                current_relation_body_instances = relation_body_instances[(relation_body_instances['file_name'] == current_image_file)]

                # get pubtator genes and coocurrences for corresponding article
                
                # fix this **************************
                # article_gene_list = article_pd.loc[(article_pd['figid'] == image_name+ext)]['gene_list'].values[0]
                article_gene_list = article_pd



                # current_pmcid = article_pd.loc[(article_pd['figid'] == image_name+ext)]['pmcid'].values[0]
                # gene_co_occurrence = pd.read_csv('nlp_pipeline_v2/from_PMCID_to_gene_annotation_and_cooccurrence/gene_co_occurrence/' + str(current_pmcid) + ".csv")

                # get ocr result
                img_id = current_element_instances['image_id'].values[0]
                processed_el_body_instances = get_ocr(current_image_file,article_gene_list,gene_name_list,data_folder,image_name,current_relation_body_instances,img_id, current_element_instances)

                # get relation head & tail
                processed_el_body_instances = get_relationship_head(processed_el_body_instances,current_relation_head_instances)
                processed_el_body_instances = get_relationship_tail(current_image_file,processed_el_body_instances)

                # remove relationships with None head or tail
                processed_el_body_instances = filter_heads_and_tails(processed_el_body_instances)

                # get gene stators and receptors
                processed_el_body_instances = get_startors_and_receptors(current_image_file,processed_el_body_instances)

                # score relationships
                # processed_el_body_instances = score_by_cooccurrence(gene_co_occurrence,processed_el_body_instances)

                # visualize elements and relationships
                save_visual(current_image_file, processed_el_body_instances,visuals_folder, image_name, ext)
                # save_visual(current_image_file, current_element_instances,visuals_folder, image_name, ext)

                

                # save outputs
                result = processed_el_body_instances[processed_el_body_instances['category_id'] != 1]
                results = result[
                    ["image_id", "file_name", "category_id", "bbox", "normalized_bbox", "startor", "startor_bbox", "relation_category",
                    "receptor","receptor_bbox"]]
                relation_path = data_folder + 'relation/'
                if not os.path.exists(relation_path):
                    os.makedirs(relation_path)
                with open('{:s}_relation.json'.format(os.path.join(relation_path, image_name)), 'w') as output_fp:
                    results.to_json(output_fp, orient='index')

                

                del current_element_instances


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # args.dataset = r'selective_figures_for_validation_set/img/group1'
    # args.dataset = r'NSCLC_pathway'

    # args.dataset = r'filtered_val_images'

    args.dataset = r'my_filtered_val_images'
    # # test
    # args.dataset = r'test'

    # # test1
    # args.dataset = r'test1'

    file_path = vars(args)['dataset']
    img_path = os.path.join(file_path, 'img/')
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    # remove this *****************
    article_pd = None

    run_model(cfg, article_pd, **vars(args))
