from tools.label_file import LabelFile
from tools.shape_tool import generate_rect_points
import os
import cv2
import numpy as np
import random


# generate gene rotated boxes
def generate_gene_rotated_box(entity_box):
    entity_box = np.array(entity_box, np.int32).reshape((-1, 2))
    gene_rotated_box = cv2.minAreaRect(entity_box)
    return gene_rotated_box


# generate relation bounding boxes
def generate_relation_bounding_box(entity1_box, entity2_box, entity3_box,offset = 0):
    # if len(entity1_box)==2:

    entity1_box = np.array(entity1_box, np.int32).reshape((-1, 2))
    entity2_box = np.array(entity2_box, np.int32).reshape((-1, 2))
    entity3_box = np.array(entity3_box, np.int32).reshape((-1, 2))

    # handle if entity boxes are rectangles and NOT polygons
    if entity1_box.shape[0] == 4:
        entity1_box = cv2.boxPoints(cv2.minAreaRect(entity1_box))
        entity1_box = np.int32(entity1_box)

    if entity2_box.shape[0] == 4:
        entity2_box = cv2.boxPoints(cv2.minAreaRect(entity2_box))
        entity2_box = np.int32(entity2_box)

    if entity3_box.shape[0] == 4:
        entity3_box = cv2.boxPoints(cv2.minAreaRect(entity3_box))
        entity3_box = np.int32(entity3_box)
    # else:
    #     entity2_box=cv2.rectangle(boxed_img,entity2_box[0],entity1_box[1],(0,0,255))

    cnt = np.concatenate((entity1_box, entity2_box, entity3_box),axis=0)
    rotated_box=cv2.minAreaRect(cnt)

    rect=cv2.boxPoints(rotated_box)

    rect[0][0]=rect[0][0]-offset
    rect[0][1] = rect[0][1] + offset
    rect[1][0] = rect[1][0] + offset
    rect[1][1] = rect[1][1] - offset
    rect[2][0]=rect[2][0] + offset
    rect[2][1] = rect[2][1] - offset
    rect[3][0] = rect[3][0] - offset
    rect[3][1] = rect[3][1] + offset

    #rect = np.int0(rect)

    return rotated_box,rect

def generate_compound_bounding_box(entity_box, offset = 0):

    entity_box_np_list = np.array(entity_box[0], np.int32).reshape((-1, 2))
    for i in range(1,len(entity_box)):
        entity_box_np = np.array(entity_box[i], np.int32).reshape((-1, 2))
        # handle if entity boxes are rectangles and NOT polygons
        if entity_box_np.shape[0] == 4:
            entity_box_np = cv2.boxPoints(cv2.minAreaRect(entity_box_np))
            entity_box_np = np.int32(entity_box_np)
        entity_box_np_list = np.concatenate((entity_box_np_list, entity_box_np), axis=0)

    rotated_compound_box=cv2.minAreaRect(entity_box_np_list)

    polygon=cv2.boxPoints(rotated_compound_box)

    polygon = np.int0(polygon)

    return rotated_compound_box, polygon

def append_new_shape(rect,rotated_box):
    new_shape = {}

    new_shape['line_color'] = [255, 0, 0, 128]
    new_shape['fill_color'] = None
    new_shape['shape_type'] = 'polygon'
    new_shape['points'] = rect.tolist()
    new_shape['flags'] = {}


    new_shape['rotated_box'] = []
    new_shape['rotated_box'].append(rotated_box[0][0])
    new_shape['rotated_box'].append(rotated_box[0][1])
    new_shape['rotated_box'].append(rotated_box[1][0])
    new_shape['rotated_box'].append(rotated_box[1][1])
    new_shape['rotated_box'].append(rotated_box[2])

    return new_shape


if __name__ == '__main__':


    image_path=r'/home/fei/Desktop/train_data/major/image/'
    json_path=r'/home/fei/Desktop/train_data/major/same_number_json/'
    new_label_with_box=r'/home/fei/Desktop/train_data/major/json_with_relations/'


    # offset = 5

    for json_file in os.listdir(json_path):

        file_name, file_ext = os.path.splitext(json_file)
        if file_ext != ".json":
            continue

        label = LabelFile(os.path.join(json_path, file_name + file_ext))
        img_path = os.path.join(image_path, label.imagePath)
        boxed_img = cv2.imread(img_path)

        #reset this json first
        label.reset()
        current_id_max = label.get_max_shape_id()

        # standardize the shape format
        for shape in label.shapes:
            shape['line_color'] = None
            shape['fill_color'] = None
            # print(file_name+ ' '+shape['label'])
            if len(shape['points']) == 2:
                shape['points'] = generate_rect_points(shape).tolist()
                shape['shape_type'] = 'polygon'

        for shape in label.shapes:

            offset = random.randint(0, 5)

            shape_type=LabelFile.get_shape_category(shape)

            if shape_type =='activate' or shape_type == 'inhibit':
                #shape['componuent']=[0,0]
                rotated_bounding_box = generate_gene_rotated_box(shape['points'])
                shape['rotated_box'] = []
                shape['rotated_box'].append(rotated_bounding_box[0][0])
                shape['rotated_box'].append(rotated_bounding_box[0][1])
                shape['rotated_box'].append(rotated_bounding_box[1][0])
                shape['rotated_box'].append(rotated_bounding_box[1][1])
                shape['rotated_box'].append(rotated_bounding_box[2])

                relation_annotation = LabelFile.get_shape_annotation(shape)

                if len(relation_annotation.split('|'))>1:
                    #case1: activate:12|5
                    if relation_annotation.split('|')[0].isdigit() and relation_annotation.split('|')[1].isdigit():
                        try:
                            start_index = label.get_index_with_id(relation_annotation.split('|')[0])
                            end_index = label.get_index_with_id(relation_annotation.split('|')[1])
                            rotated_box, rect= generate_relation_bounding_box(label.shapes[start_index]['points'],
                                                                              label.shapes[end_index]['points'],
                                                                              shape['points'],
                                                                              offset)
                            new_shape = append_new_shape(rect, rotated_box)
                        except Exception as e:
                            print(file_name + relation_annotation + str(e))
                            continue

                        current_id_max += 1
                        new_shape['label'] = str(current_id_max)+':'+shape_type+ '_relation:' + relation_annotation
                        # try:
                        new_shape['component']=[]
                        new_shape['component'].append(start_index)
                        new_shape['component'].append(end_index)
                        new_shape['component'].append(LabelFile.get_shape_index(shape))

                        label.shapes.append(new_shape)

                    #    continue
                    #case2:activate:[1,2,3]|5
                    elif relation_annotation.split('|')[0].find('[')>=0 and relation_annotation.split('|')[1].isdigit():

                        end_index=label.get_index_with_id(relation_annotation.split('|')[1])

                        activate_compound=relation_annotation.split('|')[0].replace('[','').replace(']','').split(',')
                        activate_entity_list=[]

                        for j in range(len(activate_compound)):
                            try:
                                # ID=find_index_with_id(int(activate_compound[j]),label.shapes)
                                compound_idx = label.get_index_with_id(int(activate_compound[j]))
                                activate_entity_list.append(label.shapes[compound_idx]['points'])

                            except Exception as e:
                                print("case2:error:"+img_path+relation_annotation+str(e))
                                activate_entity_list = []
                                break
                        if len(activate_entity_list) > 0:
                            rotated_compound_box, compound_polygon= \
                                generate_compound_bounding_box(activate_entity_list, offset)
                        else:
                            continue

                        current_id_max += 1
                        new_shape_compound=append_new_shape(compound_polygon,rotated_compound_box)
                        new_shape_compound['label'] = str(current_id_max)+':'+'compound:' + relation_annotation.split('|')[0].replace('[','').replace(']','')
                        new_shape_compound['component']=[]
                        new_shape_compound['component'].append(activate_compound)
                        label.shapes.append(new_shape_compound)

                        rotated_relation_box,relation_polygon= \
                            generate_relation_bounding_box(compound_polygon,label.shapes[end_index]['points'], shape['points'], offset)

                        current_id_max += 1
                        new_shape_relation=append_new_shape(relation_polygon,rotated_relation_box)
                        new_shape_relation['label'] = str(current_id_max)+':'+shape_type+ '_relation:'+ relation_annotation
                        new_shape_relation['component']=[]
                        new_shape_relation['component'].append(activate_compound)
                        new_shape_relation['component'].append(end_index)
                        new_shape_relation['component'].append(LabelFile.get_shape_index(shape))
                        label.shapes.append(new_shape_relation)
                        del activate_entity_list

                    # case3:activate:5|[1,2,3]
                    elif relation_annotation.split('|')[1].find('[') >= 0 and relation_annotation.split('|')[0].isdigit():
                        start_index = label.get_index_with_id(relation_annotation.split('|')[0])
                        # try:
                        #     start_idx = find_index_with_id(start_id, label.shapes)
                        # except:
                        #     print(file_name+relation)
                        receive_compound = relation_annotation.split('|')[1].replace('[', '').replace(']','').split(',')
                        receive_entity_list = []

                        for j in range(len(receive_compound)):
                            try:
                                # ID = find_index_with_id(int(receive_compound[j]), label.shapes)
                                compound_entity_idx = label.get_index_with_id(int(receive_compound[j]))
                                receive_entity_list.append(label.shapes[compound_entity_idx]['points'])

                            except Exception as e:
                                print("case3:error:"+img_path+relation_annotation + str(e))
                                receive_entity_list = []
                                break
                        if len(receive_entity_list) > 0:
                            rotated_compound_box, compound_polygon = \
                                generate_compound_bounding_box(receive_entity_list, offset)
                        else:
                            continue
                        current_id_max += 1
                        new_shape_compound=append_new_shape(compound_polygon,rotated_compound_box)
                        new_shape_compound['label'] = str(current_id_max)+':'+'compound:' + relation_annotation.split('|')[1].replace('[','').replace(']','')
                        new_shape_compound['component']=[]
                        new_shape_compound['component'].append(receive_compound)
                        label.shapes.append(new_shape_compound)

                        rotated_relation_box,relation_polygon = \
                            generate_relation_bounding_box(label.shapes[start_index]['points'],
                                                           compound_polygon, shape['points']
                                                           , offset)

                        current_id_max += 1
                        new_shape_relation = append_new_shape(relation_polygon, rotated_relation_box)
                        new_shape_relation['label'] = str(current_id_max)+':'+shape_type+ '_relation:' + relation_annotation
                        new_shape_relation['component'] = []
                        new_shape_relation['component'].append(receive_compound)
                        new_shape_relation['component'].append(start_index)
                        new_shape_relation['component'].append(LabelFile.get_shape_index(shape))
                        label.shapes.append(new_shape_relation)
                        del receive_entity_list

                    # case4:activate:[1,2,3]|[4,5]
                    elif relation_annotation.split('|')[0].find('[') >= 0 and relation_annotation.split('|')[1].find('[') >= 0:
                        activate_compound = relation_annotation.split('|')[0].replace('[', '').replace(']','').split(',')
                        activate_entity_list = []

                        receive_compound = relation_annotation.split('|')[1].replace('[', '').replace(']','').split(',')
                        receive_entity_list = []

                        for j in range(len(activate_compound)):
                            try:
                                # ID = find_index_with_id(int(activate_compound[j]), label.shapes)
                                #ID = int(activate_compound[j])
                                compound_entity_idx = label.get_index_with_id(activate_compound[j])
                                activate_entity_list.append(label.shapes[compound_entity_idx]['points'])

                            except Exception as e:
                                activate_entity_list = []
                                print("case4:error:"+img_path+relation_annotation + str(e))
                                break
                        if len(activate_entity_list) > 0:
                            rotated_compound_a_box, compound_a_polygon = \
                                generate_compound_bounding_box(activate_entity_list, offset)

                            current_id_max += 1
                            new_shape_compound_a = append_new_shape(compound_a_polygon, rotated_compound_a_box)
                            new_shape_compound_a['label'] = str(current_id_max) + ':' + 'compound:' + \
                                                            relation_annotation.split('|')[0].replace('[', '').replace(
                                                                ']', '')
                            new_shape_compound_a['component'] = []
                            new_shape_compound_a['component'].append(activate_compound)
                            label.shapes.append(new_shape_compound_a)
                        else:
                            continue

                        for j in range(len(receive_compound)):
                            try:
                                # ID = find_index_with_id(int(receive_compound[j]), label.shapes)
                                compound_entity_idx = label.get_index_with_id(receive_compound[j])
                                #ID = int(receive_compound[j])
                                receive_entity_list.append(label.shapes[compound_entity_idx]['points'])

                            except Exception as e:
                                receive_entity_list = []
                                print("case4:error:"+file_name+relation_annotation+str(e))
                        if len(receive_entity_list) > 0:
                            rotated_compound_r_box, compound_r_polygon = \
                                generate_compound_bounding_box(receive_entity_list, offset)
                            current_id_max += 1
                            new_shape_compound_r = append_new_shape(compound_r_polygon, rotated_compound_r_box)
                            new_shape_compound_r['label'] = str(current_id_max) + ':' + 'compound:' + \
                                                            relation_annotation.split('|')[1].replace('[', '').replace(
                                                                ']', '')
                            new_shape_compound_r['component'] = []
                            new_shape_compound_r['component'].append(receive_compound)
                            label.shapes.append(new_shape_compound_r)
                        else:
                            continue



                        rotated_relation_box,relation_polygon = \
                            generate_relation_bounding_box(compound_a_polygon,compound_r_polygon, shape['points']
                                                           , offset)
                        current_id_max += 1
                        new_shape_relation = append_new_shape(relation_polygon, rotated_relation_box)
                        new_shape_relation['label'] = str(current_id_max)+':'+shape_type+ '_relation:' + relation_annotation
                        new_shape_relation['component'] = []
                        new_shape_relation['component'].append(activate_compound)
                        new_shape_relation['component'].append(receive_compound)
                        new_shape_relation['component'].append(LabelFile.get_shape_index(shape))
                        label.shapes.append(new_shape_relation)
                        del receive_entity_list,activate_entity_list

            elif shape_type =='gene':
                rotated_box=generate_gene_rotated_box(shape['points'])
                shape['rotated_box'] = []
                shape['rotated_box'].append(rotated_box[0][0])
                shape['rotated_box'].append(rotated_box[0][1])
                shape['rotated_box'].append(rotated_box[1][0])
                shape['rotated_box'].append(rotated_box[1][1])
                shape['rotated_box'].append(rotated_box[2])


        label.save(os.path.join(new_label_with_box, file_name+ '.json'),
                   label.shapes,
                   label.imagePath,
                   label.imageHeight, label.imageWidth)

