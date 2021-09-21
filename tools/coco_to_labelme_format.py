import json
import os.path as osp
from label_file import LabelFile
import pandas as pd
import numpy as np
from PIL import Image

def read_coco_format_output(coco_file_name):
    with open(coco_file_name, 'r') as coco_fp:
        predictions_in_coco_format = json.load(coco_fp)

    return predictions_in_coco_format


def convert_coco_to_labelme_format(save_folder, image_folder, predictions_in_coco):
    df_coco_predictions = pd.DataFrame(predictions_in_coco)
    image_num =  df_coco_predictions['image_id'].max() + 1
    for img_idx in range(0, image_num):
        current_img_predictions = df_coco_predictions[(df_coco_predictions['image_id'] == img_idx)
                                                      & (df_coco_predictions['score'] >= 0.7)]
        json_file_name = osp.basename(current_img_predictions['file_name'].iloc[0])

        current_image = Image.open(osp.join(image_folder, json_file_name))

        #parse predictive shapes
        current_shapes = []
        for shape_idx, each_predictions in current_img_predictions.iterrows():
            shape = format_each_predict_shape(shape_idx, each_predictions)
            current_shapes.append(shape)
            del shape

        #save predictive shapes
        json_file = LabelFile()
        json_file.save(osp.join(save_folder, json_file_name[:-4]+'.json'),
                       shapes=current_shapes,imagePath=json_file_name,
                       imageWidth= current_image.width, imageHeight= current_image.height)
        del current_img_predictions,current_image, current_shapes


def format_each_predict_shape(shape_idx, each_prediction):
    tempDict = {}
    category_id = each_prediction['category_id']

    left_x, left_y, width, height = each_prediction['bbox']

    if category_id == 0:
        tempDict['label'] = str(shape_idx) + ':activate:'
        tempDict['line_color'] = [255, 0, 0, 128]
        tempDict['fill_color'] = None
        tempDict['points'] = [[left_x, left_y],
                              [left_x + width, left_y + height]]
        tempDict['shape_type'] = 'rectangle'

    elif category_id == 1:
        tempDict['label'] = str(shape_idx) + ':gene:'
        tempDict['line_color'] = [0, 0, 0, 128]
        tempDict['fill_color'] = None
        tempDict['points'] = [[left_x , left_y ],
                              [left_x + width, left_y + height]]
        tempDict['shape_type'] = 'rectangle'
    elif category_id == 2:
        tempDict['label'] = str(shape_idx) + ':inhibit:'
        tempDict['line_color'] = [0, 255, 0, 128]
        tempDict['fill_color'] = None
        tempDict['points'] = [[left_x , left_y ],
                              [left_x + width, left_y + height]]
        tempDict['shape_type'] = 'rectangle'
    return tempDict

if __name__ == "__main__":
    coco_file_name = r'C:\Users\coffe\Desktop\pathway_inference\pathway_val_0_regular_element\coco_instances_results.json'
    save_folder = r'C:\Users\coffe\Desktop\pathway_inference\pathway_val_0_regular_element\jsons'
    image_folder = r'C:\Users\coffe\Desktop\pathway_inference\img+json'
    predictions = read_coco_format_output(coco_file_name)
    convert_coco_to_labelme_format(save_folder, image_folder,predictions)