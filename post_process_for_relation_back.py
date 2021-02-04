import os
from tqdm import tqdm
from tools.shape_tool import relation_covers_this_element
import json
import pandas as pd
import numpy as np
import cv2
import cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from train_net import get_cfg, Trainer, RegularTrainer
from detectron2.checkpoint import DetectionCheckpointer
from pathway_evaluation import PathwayEvaluator, RegularEvaluator
from tools.relation_data_tool import register_Kfold_pathway_dataset
from OCR import OCR

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
        return np.array([[pt0, pt1], [pt2, pt3], [pt4, pt5], [pt6, pt7]], np.int32).reshape((4, 2))
    if len(points) == 5:
        cnt_x, cnt_y, w, h, angle = points
        return np.array(cv2.boxPoints(((cnt_x, cnt_y),(w, h), angle)), np.int32).reshape((4, 2))

def normalize_all_boxes(prediction_instances, image_size):
    assert 'normalized_bbox' in  prediction_instances.columns.values

    prediction_instances['normalized_bbox'] = prediction_instances['bbox'].map(normalize_rect_vertex, arg=(image_size,))


def get_element_rowid(element_instances, element_info_series):
    return int(element_instances.loc[(element_instances['image_id'] == element_info_series['image_id'])&
                          (element_instances['category_id'] == element_info_series['category_id'])&
                          (element_instances['score'] == element_info_series['score'])].index.values[0])

def visualize_element_and_relation(datasetDict, element_instances, relation_instances, element_threshold, relation_threshold,
                                   save_folder):


    for sample_info in tqdm(datasetDict):
        element_instances_on_sample = element_instances.loc[(element_instances['image_id'] == sample_info['image_id']) &
                                                            (element_instances['score'] >= element_threshold)]
        relation_instances_on_sample = relation_instances.loc[(relation_instances['image_id'] ==
                                                              sample_info['image_id']) &
                                                              ((relation_instances['score'] >= relation_threshold))]
        img = cv2.imread(sample_info['file_name'])
        for element_idx in range(0, len(element_instances_on_sample)):
            cv2.polylines(img, [element_instances_on_sample.iloc[element_idx]['normalized_bbox']], isClosed= True, color= [255,0,0], thickness= 2 )
        for relation_idx in range(0, len(relation_instances_on_sample)):
            cv2.polylines(img, [relation_instances_on_sample.iloc[relation_idx]['normalized_bbox']], isClosed=True, color=[0, 255, 0], thickness=2)

        cv2.imwrite(os.path.join(save_folder, os.path.basename(sample_info['file_name'])), img)
        del  img


def filter_invalid_relation_predictions(relation_prediction_file,datasetDict, element_instances, relation_instances,
                                        element_threshold, cover_ratio):

    assert 'normalized_bbox' in element_instances.columns.values
    assert 'normalized_bbox' in relation_instances.columns.values

    valid_relations = []
    for sample_info in tqdm(datasetDict):
        element_instances_on_sample = element_instances.loc[(element_instances['image_id'] == sample_info['image_id'])&
                                                              (element_instances['score'] >= element_threshold)]
        relation_instances_on_sample = relation_instances.loc[relation_instances['image_id'] ==
                                sample_info['image_id']]

        for relation_idx in range(len(relation_instances_on_sample)):
            current_relation_vertex = relation_instances_on_sample.iloc[relation_idx]['normalized_bbox']
            #current_relation_vertex = normalize_rect_vertex(current_relation_vertex)
            current_relation_category = cfg.relation_list[relation_instances_on_sample.iloc[relation_idx]['category_id']]
            genes_in_relation_count = 0
            relation_symbol_count = 0
            relation_symbol = False
            gene_counts = False
            covered_elements = []
            for element_idx in range(len(element_instances_on_sample)):
                current_element_vertex = element_instances_on_sample.iloc[element_idx]['normalized_bbox']
                #current_element_vertex = normalize_rect_vertex(current_element_vertex)
                current_element_category = cfg.element_list[element_instances_on_sample.iloc[element_idx]['category_id']]
                if  relation_covers_this_element(element_box_points= current_element_vertex, relation_box_points= current_relation_vertex, cover_ratio= cover_ratio):
                    if  current_element_category == 'gene':
                        genes_in_relation_count += 1
                        #get the covered element's index
                        covered_elements.append(element_idx)

                        if genes_in_relation_count >= 2:
                            gene_counts = True
                        else:
                            gene_counts = False
                    else:
                        relation_symbol_count += 1
                        covered_elements.append(element_idx)
                        if relation_symbol_count > 0: #and (current_relation_category.find(current_element_category) != -1):
                            relation_symbol = True
            valid_relation_instance = relation_instances_on_sample.iloc[relation_idx].copy()
            valid_relation_instance['covered_elements'] = covered_elements
            del covered_elements
            if relation_symbol and gene_counts:
                # get all valid_relations
                valid_relations.append(valid_relation_instance.to_dict())

    for valid_relation in valid_relations:
        valid_relation['image_id'] = int (valid_relation['image_id'])
        valid_relation['category_id'] = int (valid_relation['category_id'])
        valid_relation['score'] = float(valid_relation['score'])
        # valid_relation['normalized_bbox'] = valid_relation['normalized_bbox'].reshape(-1).tolist()
        del valid_relation['normalized_bbox']

    with open(relation_prediction_file[:-5]+'_new.json', "w") as f:
        f.write(json.dumps(valid_relations))
        f.flush()

    #return pd.DataFrame(valid_relations)


if __name__ == "__main__":
    # import the relation_retinanet as meta_arch, so they will be registered
    from relation_retinanet import RelationRetinaNet
    #import gene name list as ocr user_words
    with open(cfg.dictionary_path) as gene_name_list_fp:
        gene_name_list = json.load(gene_name_list_fp)

    #register data
    img_path = r'/home/fei/Desktop/weiwei/data/20200309/images/'
    json_path = r'/home/fei/Desktop/weiwei/data/20200309/jsons/'

    register_Kfold_pathway_dataset(json_path, img_path, cfg.relation_list, K =1)
    # register_Kfold_pathway_dataset(json_path, img_path, element_list, K=1)


    #run element prediction
    element_cfg = get_cfg()
    element_cfg.merge_from_file(r'./Base-RetinaNet.yaml')
    element_cfg.OUTPUT_DIR = r'./output/element/'
    element_cfg.freeze()

    # element_model = RegularTrainer.build_model(element_cfg)
    #
    # DetectionCheckpointer(model= element_model,
    #                       save_dir=element_cfg.OUTPUT_DIR).resume_or_load(
    #                       os.path.join(element_cfg.OUTPUT_DIR, 'element_model.pth'), resume=False)
    # element_evaluation_res = RegularTrainer.test(element_cfg,element_model, RegularEvaluator(element_cfg.DATASETS.TEST[0],element_cfg,True,False, element_cfg.OUTPUT_DIR))
    #
    # del element_cfg,element_model


    #run relation prediction
    relation_cfg = get_cfg()
    relation_cfg.merge_from_file(r'./Base-RetinaNet-relation.yaml')
    relation_cfg.OUTPUT_DIR = r'./output/relation/'
    relation_cfg.freeze()


    relation_model = RegularTrainer.build_model(relation_cfg)
    DetectionCheckpointer(model=relation_model,
                          save_dir=relation_cfg.OUTPUT_DIR).resume_or_load(
                           os.path.join(relation_cfg.OUTPUT_DIR, 'model_0017906.pth'), resume=False)
    relation_evaluation_res =  RegularTrainer.test(relation_cfg, relation_model,
                     RegularEvaluator(relation_cfg.DATASETS.TEST[0], relation_cfg,
                                      True, False, relation_cfg.OUTPUT_DIR))
    del relation_model



    #run rotated-relation prediction
    rotated_relation_cfg = get_cfg()
    rotated_relation_cfg.merge_from_file(r'./Base-RelationRetinaNet.yaml')
    rotated_relation_cfg.OUTPUT_DIR = r'./output/rotated_relation/'
    rotated_relation_cfg.freeze()

    # rotated_relation_model = Trainer.build_model(rotated_relation_cfg)
    # DetectionCheckpointer(model=rotated_relation_model,
    #                       save_dir=rotated_relation_cfg.OUTPUT_DIR).resume_or_load(
    #                        os.path.join(rotated_relation_cfg.OUTPUT_DIR, 'relation_model.pth'), resume=False)
    # rotated_relation_evaluation_res =  Trainer.test(rotated_relation_cfg, rotated_relation_model,
    #                  PathwayEvaluator(rotated_relation_cfg.DATASETS.TEST[0], rotated_relation_cfg,
    #                                   True, False, rotated_relation_cfg.OUTPUT_DIR))
    # del rotated_relation_model



    #post relation process
    datasetDict = DatasetCatalog.get(relation_cfg.DATASETS.TEST[0])


    element_prediction_file = os.path.join(element_cfg.OUTPUT_DIR, r'coco_instances_results.json')
    relation_prediction_file = os.path.join(relation_cfg.OUTPUT_DIR, r'coco_instances_results.json')
    rotated_relation_prediction_file = os.path.join(rotated_relation_cfg.OUTPUT_DIR, r'coco_instances_results.json')
    # datasetDict = DatasetCatalog.get(rotated_relation_cfg.DATASETS.TEST[0])
    element_predictions = json.load(open(element_prediction_file, 'r'))
    relation_predictions = json.load(open(relation_prediction_file, 'r'))
    rotated_relation_predictions = json.load(open(rotated_relation_prediction_file, 'r'))
    element_instances = pd.DataFrame(element_predictions)
    relation_instances = pd.DataFrame(relation_predictions)
    rotated_relation_instances = pd.DataFrame(rotated_relation_predictions)
    # entend pairing gene column to relation_instances for saving paired element ids
    relation_instances['covered_elements'] = None
    relation_instances['normalized_bbox'] = None
    rotated_relation_instances['covered_elements'] = None
    rotated_relation_instances['normalized_bbox'] = None
    element_instances['normalized_bbox'] = None

    #normalize all boxes into rectangle
    normalize_all_boxes(rotated_relation_instances)
    normalize_all_boxes(relation_instances)
    normalize_all_boxes(element_instances)

    #may delete if found better strategy
    filter_invalid_relation_predictions(rotated_relation_prediction_file,datasetDict, element_instances, rotated_relation_instances,
                                        cfg.element_threshold, cover_ratio=0.3)
    # filter_invalid_relation_predictions(relation_prediction_file,datasetDict, element_instances, element_list, relation_instances,
    #                                     relation_list, cfg.element_threshold, cover_ratio=0.3)

    #load new filtered relation_predictions
    del rotated_relation_instances,relation_instances

    relation_predictions = json.load(open(relation_prediction_file[:-5] + '_new.json', 'r'))
    relation_instances = pd.DataFrame(relation_predictions)
    relation_instances['normalized_bbox'] = None
    normalize_all_boxes(relation_instances)

    rotated_relation_predictions = json.load(open(rotated_relation_prediction_file[:-5] + '_new.json', 'r'))
    rotated_relation_instances = pd.DataFrame(rotated_relation_predictions)
    rotated_relation_instances['normalized_bbox'] = None
    normalize_all_boxes(rotated_relation_instances)
    #re-evaluate relation prediction
    # evaluator = PathwayEvaluator(relation_cfg.DATASETS.TEST[0], relation_cfg, True, False, relation_cfg.OUTPUT_DIR)
    # evaluator.reset()
    # evaluator.read_predictions_with_coco_format_from_json_file(os.path.join(relation_cfg.OUTPUT_DIR, 'coco_instances_results_new.json'))
    # evaluator.evaluate()
    # del relation_cfg


    # visualize_element_and_relation(datasetDict, element_instances, relation_instances,
    #                                cfg.element_threshold, cfg.relation_threshold,
    #                                r'/home/fei/Desktop/weiwei/data/20200309/visualize/relation/')
    #
    # visualize_element_and_relation(datasetDict, element_instances, rotated_relation_instances,
    #                                cfg.element_threshold, cfg.relation_threshold,
    #                                r'/home/fei/Desktop/weiwei/data/20200309/visualize/rotated_relation/')


    # extend ocr result column to element_instances for saving ocr results
    element_instances['ocr'] = None
    from OCR import ocr_text_from_image
    from predict_relationship import get_relationship_pairs_on_single_image
    from formulate_relation import get_gene_pairs_on_relation_sub_image, generate_sub_image_bounding_relation_rotated,generate_sub_image_bounding_relation_regular

    relation_instances['pair_elements'] = None
    # for sample in datasetDict:
    #
    #     # do OCR
    #     # results, all_results_dict, corrected_results_dict, fuzz_ratios_dict, coordinates_list  = \
    #     #     OCR(sample['file_name'], relation_cfg.DATASETS.TEST[0], cfg.predict_folder,
    #     #                  current_elements.loc[current_elements['category_id'] == element_list.index('gene')])
    #
    #
    #     # do gene pairing
    #     predicted_relationship_pairs, pair_descriptions, predicted_relationship_boxes = get_relationship_pairs_on_single_image(
    #         sample['file_name'],
    #         element_instances.loc[(sample['image_id'] == element_instances['image_id']) &
    #                               (element_instances['score'] >= cfg.element_threshold)],
    #         relation_instances.loc[(sample['image_id'] == relation_instances['image_id']) &
    #                                (relation_instances['score'] >= cfg.relation_threshold)])
    #
    #     #visualize pairing results
    #
    # del element_instances, relation_instances, element_predictions, relation_predictions, datasetDict
    for sample in datasetDict:
        element_instances_on_sample = element_instances.loc[(sample['image_id'] == element_instances['image_id']) &
                                            (element_instances['score'] >= cfg.element_threshold)]
                                            #&(element_instances['category_id'] == element_list.index('gene'))]

        relation_symbol_instances_on_sample = element_instances.loc[(sample['image_id'] == element_instances['image_id']) &
                                            (element_instances['score'] >= cfg.element_threshold) &
                                            (element_instances['category_id'] != cfg.element_list.index('gene'))]


        relation_instances_on_sample = relation_instances.loc[(sample['image_id'] == relation_instances['image_id']) &
                                                              (relation_instances['score'] >= cfg.relation_threshold)]

        # relation_instances_on_sample = rotated_relation_instances.loc[(sample['image_id'] == rotated_relation_instances['image_id']) &
        #                                                       (rotated_relation_instances['score'] >= cfg.rotated_relation_threshold)]
      #  # do OCR
        # all_results_dict, corrected_results_dict, fuzz_ratios_dict  = \
        #     OCR(sample['file_name'], cfg.sub_image_folder_for_ocr, element_instances_on_sample,
        #         element_list, user_words=gene_name_list)

        #TODO pick best results to 'ocr' column



        # pair involving entities relation by relation
        img = cv2.imread(sample['file_name'])
        image_name, image_ext = os.path.splitext(os.path.basename(sample['file_name']))



        for relation_index in range(0, len(relation_instances_on_sample)):
            # #rotated bbox
            sub_img, covered_element_instances = generate_sub_image_bounding_relation_rotated(img,
                                                 relation_instances_on_sample.iloc[relation_index],
                                                 element_instances_on_sample,
                                                 1)

            #plot elements on sub_img using their perspective bbox
            # for element_idx in range(0, len(covered_element_instances)):
            #     cv2.polylines(sub_img, [covered_element_instances.iloc[element_idx]['perspective_bbox']], isClosed=True,
            #                   color=[255, 0, 0], thickness=2)
            # save sub-image to visualize sub-img
            # cv2.imwrite(os.path.join(r'/home/fei/Desktop/vis_results_old/normalized/',
            #                          image_name + str(relation_index) + image_ext), sub_img)


            # #regular bbox
            # sub_img, covered_element_instances = generate_sub_image_bounding_relation_regular(img,
            #                                      relation_instances_on_sample.iloc[relation_index],
            #                                      element_instances_on_sample,
            #                                      1)


            #plot elements on sub_img using their perspective bbox
            for element_idx in range(0, len(covered_element_instances)):
                cv2.polylines(sub_img, [covered_element_instances.iloc[element_idx]['perspective_bbox']], isClosed=True,
                              color=[255, 0, 0], thickness=2)
            # save sub-image to visualize sub-img
            cv2.imwrite(os.path.join(r'/home/fei/Desktop/vis_results_old/normalized/',
                                     image_name + str(relation_index) + image_ext), sub_img)

            # do gene pairing
            detected_relation_info = get_gene_pairs_on_relation_sub_image(
                sub_img, element_instances_on_relation= covered_element_instances, image_name= image_name,
                image_ext= image_ext, idx = relation_index, )

            #visualize pairing results

            del sub_img, covered_element_instances

        del img

    del element_instances, relation_instances, element_predictions, relation_predictions, datasetDict