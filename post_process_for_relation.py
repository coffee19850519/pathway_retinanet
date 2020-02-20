import os
from tqdm import tqdm
from tools.shape_tool import relation_covers_this_element
import json
import pandas as pd
import numpy as np
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from train_net import get_cfg, Trainer
from detectron2.checkpoint import DetectionCheckpointer
from pathway_evaluation import PathwayEvaluator
from tools.relation_data_tool import register_Kfold_pathway_dataset


def normalize_rect_vertex(points):
    if len(points) == 4:
        boxes = np.array(points, np.float).reshape((-1, 4))
        boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        points = np.array(boxes).reshape((2, 2))
        pt0 = np.min(points[:, 0])
        pt1 = np.min(points[:, 1])
        pt4 = np.max(points[:, 0])
        pt5 = np.max(points[:, 1])
        pt2 = pt4
        pt3 = pt1
        pt6 = pt0
        pt7 = pt5
        del points, boxes
        return np.array([[pt0, pt1], [pt2, pt3], [pt4, pt5], [pt6, pt7]]).reshape((4, 2))
    if len(points) == 5:
        cnt_x, cnt_y, w, h, angle = points
        return np.array(cv2.boxPoints(((cnt_x, cnt_y),(w, h), angle))).reshape((4, 2))

def filter_invalid_relation_predictions(dataset_name, element_prediction_file, element_list, relation_prediction_file, relation_list,
                                        element_threshold, cover_ratio):
    datasetDict = DatasetCatalog.get(dataset_name)
    element_predictions = json.load(open(element_prediction_file, 'r'))
    relation_predictions = json.load(open(relation_prediction_file, 'r'))
    element_instances = pd.DataFrame(element_predictions)
    relation_instances = pd.DataFrame(relation_predictions)
    valid_relations = []
    for sample_info in tqdm(datasetDict):
        element_instances_on_sample = element_instances.loc[(element_instances['image_id'] == sample_info['image_id'])&
                                                              (element_instances['score'] >= element_threshold)]
        relation_instances_on_sample = relation_instances.loc[relation_instances['image_id'] ==
                                sample_info['image_id']]

        for relation_idx in range(len(relation_instances_on_sample)):
            current_relation_vertex = relation_instances_on_sample.iloc[relation_idx]['bbox']
            current_relation_vertex = normalize_rect_vertex(current_relation_vertex)
            current_relation_category = relation_list[relation_instances_on_sample.iloc[relation_idx]['category_id']]
            genes_in_relation_count = 0
            relation_symbol = False
            gene_counts = False

            for element_idx in range(len(element_instances_on_sample)):
                current_element_vertex = element_instances_on_sample.iloc[element_idx]['bbox']
                current_element_vertex = normalize_rect_vertex(current_element_vertex)
                current_element_category = element_list[element_instances_on_sample.iloc[element_idx]['category_id']]
                if relation_covers_this_element(current_relation_vertex, current_element_vertex, cover_ratio):
                        if current_element_category == 'gene':
                            genes_in_relation_count += 1
                            if genes_in_relation_count >= 2:
                                gene_counts = True
                                continue
                            else:
                                gene_counts = False
                                continue
                        else:
                            if current_relation_category.find(current_element_category) != -1:
                                #TODO:how to keep it in the following symbol testing?
                                relation_symbol = True
                                continue
            if relation_symbol and gene_counts:
                # get all valid_relations
                valid_relations.append(relation_instances_on_sample.iloc[relation_idx].to_dict())

    for valid_relation in valid_relations:
        valid_relation['image_id'] = int (valid_relation['image_id'])
        valid_relation['category_id'] = int (valid_relation['category_id'])
        valid_relation['score'] = float(valid_relation['score'])

    with open(relation_prediction_file[:-5]+'_new.json', "w") as f:
        f.write(json.dumps(valid_relations))
        f.flush()
    del valid_relations, element_instances,relation_instances,element_predictions,relation_predictions,datasetDict

if __name__ == "__main__":
    # import the relation_retinanet as meta_arch, so they will be registered
    from relation_retinanet import RelationRetinaNet

    #register data
    img_path = r'/home/fei/Desktop/test/images/'
    json_path = r'/home/fei/Desktop/test/jsons/'
    element_list = ['activate', 'gene', 'inhibit']
    relation_list = ['activate_relation', 'inhibit_relation']
    register_Kfold_pathway_dataset(json_path, img_path, relation_list, K=1)
    #run element prediction
    element_cfg = get_cfg()
    element_cfg.merge_from_file(r'./Base-RetinaNet.yaml')
    element_cfg.OUTPUT_DIR = r'./output/element/'
    element_cfg.freeze()
    #
    # element_model = Trainer.build_model(element_cfg)
    #
    # DetectionCheckpointer(model= element_model,
    #                       save_dir=element_cfg.OUTPUT_DIR).resume_or_load(
    #                       os.path.join(element_cfg.OUTPUT_DIR, 'element_model.pth'), resume=False)
    # element_evaluation_res = Trainer.test(element_cfg,element_model, COCOEvaluator(element_cfg.DATASETS.TEST[0],element_cfg,True,element_cfg.OUTPUT_DIR))
    #
    # del element_cfg,element_model
    #run relation prediction
    relation_cfg = get_cfg()
    relation_cfg.merge_from_file(r'./Base-RelationRetinaNet.yaml')
    relation_cfg.OUTPUT_DIR = r'./output/relation/'
    relation_cfg.freeze()
    # relation_model = Trainer.build_model(relation_cfg)
    #
    # DetectionCheckpointer(model=relation_model,
    #                       save_dir=relation_cfg.OUTPUT_DIR).resume_or_load(
    #                        os.path.join(relation_cfg.OUTPUT_DIR, 'relation_model.pth'), resume=False)
    # relation_evaluation_res = Trainer.test(relation_cfg, relation_model,  PathwayEvaluator(relation_cfg.DATASETS.TEST[0], relation_cfg, True,False, relation_cfg.OUTPUT_DIR))
    # del relation_cfg, relation_model


    #post relation process
    element_prediction_file = os.path.join(element_cfg.OUTPUT_DIR, r'coco_instances_results.json')
    relation_prediction_file = os.path.join(relation_cfg.OUTPUT_DIR, r'coco_instances_results.json')
    filter_invalid_relation_predictions(relation_cfg.DATASETS.TEST[0], element_prediction_file, element_list, relation_prediction_file,
                                        relation_list, 0.85, 0.05)
    #re-evaluate relation prediction


