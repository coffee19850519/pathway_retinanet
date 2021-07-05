# -*- coding: utf-8 -*-
from __future__ import print_function
from google.cloud import vision_v1 as vision
# from google.cloud.vision.vision import types
import io,os, json
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'Pathway-3f29d8393d4b.json'
import numpy as np
from detectron2.structures import BoxMode,Boxes
import cv2

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


# image_uri = 'gs://cloud-vision-codelab/otter_crossing.jpg'
def gcv_ocr(file_name,current_element_instances):

    img = cv2.imread(file_name)
    height, width, _ = img.shape
    image_size = (height,width)

    # path = '/home/fei/Desktop/weiwei/pathway_web/SkyEye/users/upload-files/test_img/cin_00085.jpg'
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    client = vision.ImageAnnotatorClient()
    # image = vision.types.Image()
    # image.source.image_uri = image_uri
    image = vision.types.Image(content=content)

    # has xywh
    model_bboxes_temp = current_element_instances.bbox.values
    model_bboxes = []
    for box in model_bboxes_temp:
        model_bboxes.append(list(map(int,box)))
    model_scores = [0.9] * len(model_bboxes)

    

    # returns xyxy
    response = client.text_detection(image=image, image_context = {"language_hints" : ["en"]})

    # get just words
    results=[]
    google_bboxes=[]
    for text in response.text_annotations:
        vertices = [[v.x, v.y] for v in text.bounding_poly.vertices]

        

        x1 = vertices[0][0]
        y1 = vertices[0][1]
        x2 = vertices[2][0]
        y2 = vertices[2][1]
        temp_bbox =  BoxMode.convert(np.array([x1,y1,x2,y2]).reshape((-1, 4)),BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).tolist()[0]
        google_bboxes.append(temp_bbox)

    google_scores = [0.5] *len(google_bboxes)

    bboxes = model_bboxes + google_bboxes
    scores = model_scores + google_scores

    # xywh_boxes = []
    # for box in bboxes:
    #     temp_bbox = BoxMode.convert(np.array(box).reshape((-1, 4)),BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).tolist()[0]
    #     xywh_boxes.append(temp_bbox)


    # just doing nms still runs into issue cutting off pieces and adding extra
    # TODO:: try combination step next
    filtered_bboxes_indxs = cv2.dnn.NMSBoxes(bboxes,scores,0.8,0.9)
    filtered_bboxes_indxs = [int(index) for index_list in filtered_bboxes_indxs for index in index_list]

    print(filtered_bboxes_indxs)

    filtered_bboxes = list(bboxes[i] for i in filtered_bboxes_indxs)
    for box in filtered_bboxes:
        cv2.polylines(img,[normalize_rect_vertex(box, image_size)], isClosed=True, color=(0, 0, 255), thickness=2)

    image_name, ext = os.path.splitext(os.path.basename(file_name))
    cv2.imwrite(os.path.join(image_name + ext), img)













    


    return results, filtered_bboxes



if __name__ == "__main__":
    gcv_ocr('/home/fei/Desktop/weiwei/pathway_web/SkyEye/users/upload-files/test_img/cin_00085.jpg')