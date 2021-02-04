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
from detectron2.structures import BoxMode, Boxes
from train_net import get_cfg, Trainer, RegularTrainer
from detectron2.checkpoint import DetectionCheckpointer
from pathway_evaluation import PathwayEvaluator, RegularEvaluator
from tools.relation_data_tool import register_Kfold_pathway_dataset
import random
from formulate_relation import get_gene_pairs_on_relation_sub_image, generate_sub_image_bounding_relation_rotated, \
    generate_sub_image_bounding_relation_regular


def normalize_rect_vertex(points, image_size):
    if len(points) == 4:
        boxes = np.array(points, np.float).reshape((-1, 4))
        boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        boxes = Boxes(boxes)
        boxes.clip(image_size)
        points = np.array(boxes.tensor).reshape((2, 2))
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

def normalize_all_boxes(prediction_instances,image_size,  current_image_file, threshold):
    assert 'normalized_bbox' in  prediction_instances.columns.values
    current_idx_list = (prediction_instances[(prediction_instances['file_name'] == current_image_file)
                                            & (prediction_instances['score'] >= threshold)]).index.tolist()
    for row_idx in current_idx_list:
        #locate current row at all_instances
        prediction_instances._set_value(row_idx,
                                        'normalized_bbox',
                                        normalize_rect_vertex(prediction_instances.iloc[row_idx]['bbox'], image_size))
    # prediction_instances['normalized_bbox'] = prediction_instances['bbox'].map(normalize_rect_vertex)


def get_element_rowid(element_instances, element_info_series):
    return int(element_instances.loc[(element_instances['image_id'] == element_info_series['image_id'])&
                          (element_instances['category_id'] == element_info_series['category_id'])&
                          (element_instances['score'] == element_info_series['score'])].index.values[0])

def visualize_element_and_relation(datasetDict, element_instances, relation_instances, element_threshold, relation_threshold,
                                   image_path, save_folder):


    for sample_info in tqdm(datasetDict):
        element_instances_on_sample = element_instances.loc[(element_instances['file_name'] ==
           '/home/fei/Desktop/weiwei/pathway/pathway_web/SkyEye/users/upload-files/test_img/' + os.path.basename(sample_info['file_name'])) &
                                                            (element_instances['score'] >= element_threshold)]
        relation_instances_on_sample = relation_instances.loc[(relation_instances['file_name'] ==
           '/home/fei/Desktop/weiwei/pathway/pathway_web/SkyEye/users/upload-files/test_img/' + os.path.basename(sample_info['file_name'])) &
                                                              ((relation_instances['score'] >= relation_threshold))]
        img = cv2.imread( image_path + os.path.basename(sample_info['file_name']))
        # for element_idx in range(0, len(element_instances_on_sample)):
        #     element_bbox = np.array(element_instances_on_sample.iloc[element_idx]['normalized_bbox'], np.int).reshape((4,2))
        #     cv2.polylines(img, [element_bbox], isClosed= True, color= [255,0,0], thickness= 2 )
        for relation_idx in range(0, len(relation_instances_on_sample)):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            relation_bbox = np.array(relation_instances_on_sample.iloc[relation_idx]['normalized_bbox'], np.int).reshape(
                (4, 2))
            cv2.polylines(img, [relation_bbox], isClosed=True, color=(b,g,r), thickness=2)
            cover_bboxes = relation_instances_on_sample.iloc[relation_idx]['covered_bboxes']
            for bbox in cover_bboxes:
                bbox = np.array(bbox, np.int).reshape((4, 2))
                cv2.polylines(img, [bbox], isClosed=True, color=(b, g, r), thickness=2)


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
            # current_relation_vertex = normalize_rect_vertex(current_relation_vertex, (sample_info['width'], sample_info['height']))
            #current_relation_category = cfg.relation_list[relation_instances_on_sample.iloc[relation_idx]['category_id']]
            genes_in_relation_count = 0
            relation_symbol_count = 0
            relation_symbol = False
            gene_counts = False
            covered_elements = []

            covered_gene_bbox = []

            for element_idx in range(len(element_instances_on_sample)):
                current_element_vertex = element_instances_on_sample.iloc[element_idx]['normalized_bbox']
                # current_element_vertex = normalize_rect_vertex(current_element_vertex, (sample_info['width'], sample_info['height']))
                current_element_category = cfg.element_list[element_instances_on_sample.iloc[element_idx]['category_id']]
                if  relation_covers_this_element(element_box_points= current_element_vertex, relation_box_points= current_relation_vertex, cover_ratio= cover_ratio):
                    if  current_element_category == 'gene':
                        genes_in_relation_count += 1
                        #get the covered element's index
                        covered_elements.append(element_idx)

                        covered_gene_bbox.append(current_element_vertex.reshape((-1)).tolist())

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
            valid_relation_instance['covered_bboxes'] = covered_gene_bbox
            del covered_elements, covered_gene_bbox
            if relation_symbol and gene_counts:
                # get all valid_relations
                valid_relations.append(valid_relation_instance.to_dict())

    for valid_relation in valid_relations:
        valid_relation['image_id'] = int (valid_relation['image_id'])
        valid_relation['category_id'] = int (valid_relation['category_id'])
        valid_relation['score'] = float(valid_relation['score'])
        valid_relation['normalized_bbox'] = valid_relation['normalized_bbox'].reshape((-1)).tolist()
        # del valid_relation['normalized_bbox']

    with open(relation_prediction_file[:-5]+'_new.json', "w") as f:
        f.write(json.dumps(valid_relations))
        f.flush()

    #return pd.DataFrame(valid_relations)


def predict(cfg_file_path,entity_type,checkpoint):
    config=get_cfg()
    config.merge_from_file(cfg_file_path)
    config.OUTPUT_DIR = os.path.join(r'./output/',entity_type,'')
    config.freeze()

    if entity_type!= 'rotated_relation':
        model = RegularTrainer.build_model(config)
        DetectionCheckpointer(model=model,
                              save_dir=config.OUTPUT_DIR).resume_or_load(
            os.path.join(config.OUTPUT_DIR, checkpoint), resume=False)
        evaluation_res = RegularTrainer.test(config, model,
                                             RegularEvaluator(config.DATASETS.TEST[0], config,
                                                              True, False, config.OUTPUT_DIR))
    else:
        model = Trainer.build_model(config)
        DetectionCheckpointer(model=model,
                              save_dir=config.OUTPUT_DIR).resume_or_load(
            os.path.join(config.OUTPUT_DIR, checkpoint), resume=False)
        evaluation_res = Trainer.test(config, model,
                                            PathwayEvaluator(config.DATASETS.TEST[0], config,
                                                                       True, False, config.OUTPUT_DIR))
    del model
    return config


def post_predict(OUTPUT_DIR,entity_type, image_path):
    #post relation process
    # datasetDict = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    prediction_file = os.path.join(OUTPUT_DIR, r'coco_instances_results.json')
    predictions = json.load(open(prediction_file, 'r'))
    instances = pd.DataFrame(predictions)
    # entend pairing gene column to relation_instances for saving paired element ids
    instances['normalized_bbox'] = None
    if entity_type!= 'element':
        instances['covered_elements'] = None
        instances['covered_bboxes'] = None
    if entity_type=='element':
        threshold=cfg.element_threshold
    if entity_type=='relation':
        threshold=cfg.relation_threshold
    if entity_type=='rotated_relation':
        threshold=cfg.rotated_relation_threshold

    image_file_list = set(instances['file_name'])
    print(image_file_list)
    for current_image_file in image_file_list:        
        img = cv2.imread(image_path + os.path.basename(current_image_file))
        height, width, _ = img.shape
        normalize_all_boxes(instances, (height, width), current_image_file, threshold)
        del img

    del predictions
    return prediction_file,instances

def visualize_prediction(relation_prediction_file,element_instances,datasetDict,relation_type,relation_cfg,cfg,visualize_path):
    relation_predictions = json.load(open(relation_prediction_file[:-5] + '_new.json', 'r'))
    relation_instances = pd.DataFrame(relation_predictions)
    # relation_instances['normalized_bbox'] = None
    # normalize_all_boxes(relation_instances)
    # #re-evaluate relation prediction
    # if relation_type=='relation':
    #     evaluator = RegularEvaluator(relation_cfg.DATASETS.TEST[0], relation_cfg, True, False, relation_cfg.OUTPUT_DIR)
    # elif relation_type =='rotated_relation':
    #     evaluator = PathwayEvaluator(relation_cfg.DATASETS.TEST[0], relation_cfg, True, False, relation_cfg.OUTPUT_DIR)
    # evaluator.reset()
    # evaluator.read_predictions_with_coco_format_from_json_file(os.path.join(relation_cfg.OUTPUT_DIR, 'coco_instances_results_new.json'))
    # evaluator.evaluate()
    # del relation_cfg
    visualize_element_and_relation(datasetDict, element_instances, relation_instances,
                                   cfg.element_threshold, cfg.relation_threshold,
                                   visualize_path)
    del relation_predictions
    return relation_instances


def generate_relation_sub_image_and_pairing(img,image_name,image_ext,relation_instances_on_sample,element_instances_on_sample,relation_type,subimage_path):
    sub_image_path= os.path.join(subimage_path,'sub_image')
    paired_image_path=os.path.join(subimage_path,'paired')
    if not os.path.exists(sub_image_path):
        os.mkdir(sub_image_path)

    if not os.path.exists(paired_image_path):
        os.mkdir(paired_image_path)

    for relation_index in range(0, len(relation_instances_on_sample)):
        if relation_type =='relation':
            #regular bbox
            sub_img, covered_element_instances = generate_sub_image_bounding_relation_regular(img,
                                                 relation_instances_on_sample.iloc[relation_index],
                                                 element_instances_on_sample,
                                                 1)
        elif relation_type =='rotated_relation':
            # #rotated bbox
            sub_img, covered_element_instances = generate_sub_image_bounding_relation_rotated(img,
                                                  relation_instances_on_sample.iloc[relation_index],
                                                  element_instances_on_sample,
                                                  1)

        # plot elements on sub_img using their perspective bbox
        # for element_idx in range(0, len(covered_element_instances)):
        #     cv2.polylines(sub_img, [covered_element_instances.iloc[element_idx]['perspective_bbox']], isClosed=True,
        #                   color=[255, 0, 0], thickness=2)
        # save sub-image to visualize sub-img
        # cv2.imwrite(os.path.join(sub_image_path,image_name + str(relation_index) + image_ext), sub_img)
        # TODO: visualize pairing results


        # do gene pairing
        startor, receptor = get_gene_pairs_on_relation_sub_image(
            sub_img, paired_image_path, element_instances_on_relation=covered_element_instances, image_name=image_name,
            image_ext=image_ext, idx=relation_index)


        #update paired results into relation_instances_on_sample
        if startor is not None and receptor is not None:
            relation_instances_on_sample.at[relation_index, 'startor'] = startor
            relation_instances_on_sample.at[relation_index,'relation_category'] = cfg.relation_list[relation_instances_on_sample.iloc[relation_index]['category_id']]
            relation_instances_on_sample.at[relation_index, 'receptor'] = receptor

        del sub_img, covered_element_instances

    return relation_instances_on_sample

def ocr_and_pairing(element_instances,relation_instances,relation_type,datasetDict,cfg,subimage_path):
    # extend ocr result column to element_instances for saving ocr results
    element_instances['ocr'] = None
    relation_instances['pair_elements'] = None

    for sample in datasetDict:
        element_instances_on_sample = element_instances.loc[(sample['image_id'] == element_instances['image_id']) &
                                            (element_instances['score'] >= cfg.element_threshold)]
                                            # &(element_instances['category_id'] == element_list.index('gene'))]

        relation_symbol_instances_on_sample = element_instances.loc[(sample['image_id'] == element_instances['image_id']) &
                                            (element_instances['score'] >= cfg.element_threshold) &
                                            (element_instances['category_id'] != cfg.element_list.index('gene'))]


        relation_instances_on_sample = relation_instances.loc[(sample['image_id'] == relation_instances['image_id']) &
                                                              (relation_instances['score'] >= cfg.relation_threshold)]

       # # do OCR
       #  all_results_dict, corrected_results_dict, fuzz_ratios_dict  = \
       #      OCR(sample['file_name'], cfg.sub_image_folder_for_ocr, element_instances_on_sample, user_words=gene_name_list)

        #TODO pick best results to 'ocr' column



        # pair involving entities relation by relation
        img = cv2.imread(sample['file_name'])
        image_name, image_ext = os.path.splitext(os.path.basename(sample['file_name']))


        generate_relation_sub_image_and_pairing(img, image_name, image_ext, relation_instances_on_sample,
                                                element_instances_on_sample,relation_type,subimage_path)

        del img




if __name__ == "__main__":
    # import the relation_retinanet as meta_arch, so they will be registered
    from relation_retinanet import RelationRetinaNet
    #import gene name list as ocr user_words
    # with open(cfg.dictionary_path) as gene_name_list_fp:
    #     gene_name_list = json.load(gene_name_list_fp)

    #register data
    img_path = r'/mnt/data/test/img/'
    json_path = r'/mnt/data/test/json/'

    register_Kfold_pathway_dataset(json_path, img_path, cfg.element_list, K=1)   #注册element
    # register_Kfold_pathway_dataset(json_path, img_path, cfg.relation_list, K =1)



    #run element prediction
    element_cfg_file=r'./Base-RetinaNet.yaml'
    element_checkpoint='element_model.pth'
    element_cfg=predict(element_cfg_file, 'element', element_checkpoint)


    #run relation prediction
    relation_cfg_file=r'./Base-RetinaNet-relation.yaml'
    relation_checkpoint= 'model_0017906.pth'
    relation_cfg=predict(relation_cfg_file, 'relation', relation_checkpoint)

    relation_cfg = get_cfg()
    relation_cfg.merge_from_file(r'./Base-RetinaNet-relation.yaml')
    relation_cfg.freeze()

    # #run rotated-relation prediction
    # rotated_relation_cfg_file=r'./Base-RelationRetinaNet.yaml'
    # rotated_relation_checkpoint= 'relation_model.pth'
    # rotated_relation_cfg=predict(rotated_relation_cfg_file, 'rotated_relation', rotated_relation_checkpoint)


    #post relation process
    element_prediction_file,element_instances = post_predict(r'./output/element/', 'element', img_path)
    relation_prediction_file,relation_instances = post_predict(r'./output/relation', 'relation', img_path)
    # rotated_relation_prediction_file,rotated_relation_instances = post_predict(rotated_relation_cfg, 'rotated_relation')


    relation_datasetDict = DatasetCatalog.get(relation_cfg.DATASETS.TEST[0])
    # rotated_relation_datasetDict = DatasetCatalog.get(rotated_relation_cfg.DATASETS.TEST[0])
    #may delete if found better strategy
    # filter_invalid_relation_predictions(rotated_relation_prediction_file,rotated_relation_datasetDict, element_instances, rotated_relation_instances,
    #                                      cfg.element_threshold, cover_ratio=0.3)
    filter_invalid_relation_predictions(relation_prediction_file,relation_datasetDict, element_instances, relation_instances,
                                        cfg.element_threshold, cover_ratio=0.1)

    #load new filtered relation_predictions
    # del rotated_relation_instances,relation_instances
    del relation_instances



    #visualize_prediction
    relation_visualize_path=r'/home/fei/Desktop/pathway_gt/visualization/'
    # rotated_relation_visualize_path = r'/home/fei/Desktop/weiwei/kegg/visualize/rotated_relation/'
    relation_instances=visualize_prediction(relation_prediction_file, element_instances, relation_datasetDict, 'relation', relation_cfg, cfg,relation_visualize_path)
    # rotated_relation_instances=visualize_prediction(rotated_relation_prediction_file, element_instances, rotated_relation_datasetDict, 'rotated_relation', rotated_relation_cfg, cfg,rotated_relation_visualize_path)


    # OCR and pairing

    ocr_and_pairing(element_instances, relation_instances, 'relation',relation_datasetDict, cfg,
        relation_visualize_path)

    # ocr_and_pairing(element_instances, rotated_relation_instances, 'rotated_relation',rotated_relation_datasetDict, cfg,
    #     gene_name_list,rotated_relation_visualize_path)

    del element_instances, relation_instances