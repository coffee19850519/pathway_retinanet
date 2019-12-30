from label_file import LabelFile
from shape_tool import generate_rect_points
import os
from PIL import Image, ImageDraw
#from keras.preprocessing import image
import cv2
import numpy as np
import random


# generate gene rotated boxes
def generate_gene_rotated_box(entity_box):
    entity_box = np.array(entity_box, np.int32).reshape((-1, 2))
    gene_rotated_box = cv2.minAreaRect(entity_box)
    return gene_rotated_box



# generate relation bounding boxes
def generate_relation_bounding_box(entity1_box, entity2_box, entity3_box,offset):
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

    cnt = np.concatenate((entity1_box, entity2_box,entity3_box),axis=0)
    rotated_box=cv2.minAreaRect(cnt)

    rect=cv2.boxPoints(rotated_box)

    for i in range(0,4):
        for j in range(0,2):
            if rect[i][j]<0:
                rect[i][j]=0

    rect = np.int0(rect)

    # # initialize starting dimensions of sub-image
    # left_top_x = int(min(min(entity1_box[:, 0]), min(entity2_box[:, 0])))
    # left_top_y = int(min(min(entity1_box[:, 1]), min(entity2_box[:, 1])))
    # right_bottom_x = int(max(max(entity1_box[:, 0]), max(entity2_box[:, 0])))
    # right_bottom_y = int(max(max(entity1_box[:, 1]), max(entity2_box[:, 1])))



    #  # add some padding to sub-image dimensions
    # left_top_x = left_top_x - offset
    # left_top_y = left_top_y - offset
    #
    # # if dimensions with offset are out of range (negative), then set to zero
    # if (left_top_x < 0):
    #     left_top_x = 0
    # if (left_top_y < 0):
    #     left_top_y = 0
    #
    #  # if dimensions with offset are out of range (positive), then set to max edge of original image
    # right_bottom_x = right_bottom_x + offset
    # right_bottom_y = right_bottom_y + offset
    # if right_bottom_x > img.shape[1]:
    #     right_bottom_x = img.shape[1]
    # if right_bottom_y > img.shape[0]:
    #     right_bottom_y = img.shape[0]
    #
    # # check for bad created dimensions
    # if (left_top_y == right_bottom_y) or (left_top_x == right_bottom_x):
    #     return None
    # else:
    #
    #     #cv2.rectangle(img, (left_top_x, left_top_y), (right_bottom_x, right_bottom_y), (0, 0, 255), 2)


        # return img,  left_top_x, left_top_y, right_bottom_x, right_bottom_y
    return rotated_box,rect

def generate_compound_bounding_box(img, entity_box, offset):

    entity_box_np_list = np.array(entity_box[0], np.int32).reshape((-1, 2))
    for i in range(1,len(entity_box)):
        entity_box_np = np.array(entity_box[i], np.int32).reshape((-1, 2))
        # handle if entity boxes are rectangles and NOT polygons
        if entity_box_np.shape[0] == 4:
            entity_box_np = cv2.boxPoints(cv2.minAreaRect(entity_box_np))
            entity_box_np = np.int32(entity_box_np)
        entity_box_np_list = np.concatenate((entity_box_np_list, entity_box_np), axis=0)
    # initialize starting dimensions of sub-image
    left_top_x = int(min(entity_box_np_list[:, 0]))
    left_top_y = int(min(entity_box_np_list[:, 1]))
    right_bottom_x = int(max(entity_box_np_list[:, 0]))
    right_bottom_y = int(max(entity_box_np_list[:, 1]))

    # add some padding to sub-image dimensions
    left_top_x = left_top_x - offset
    left_top_y = left_top_y - offset

    # if dimensions with offset are out of range (negative), then set to zero
    if (left_top_x < 0):
        left_top_x = 0
    if (left_top_y < 0):
        left_top_y = 0

    # if dimensions with offset are out of range (positive), then set to max edge of original image
    right_bottom_x = right_bottom_x + offset
    right_bottom_y = right_bottom_y + offset
    if right_bottom_x > img.shape[1]:
        right_bottom_x = img.shape[1]
    if right_bottom_y > img.shape[0]:
        right_bottom_y = img.shape[0]



    #cv2.rectangle(img, (left_top_x, left_top_y), (right_bottom_x, right_bottom_y), (0, 0, 255), 2)
    return img, left_top_x, left_top_y, right_bottom_x, right_bottom_y






if __name__ == '__main__':



    new_labels=r'/home/fei/Desktop/new_json'
    new_label_with_box=r'/home/fei/Desktop/new_labels_with_box'


    offset = 0

    for json_file in os.listdir(new_labels):

        file_name, file_ext = os.path.splitext(json_file)
        if file_ext != ".json" :
            continue

        img_path=os.path.join(new_labels,file_name+".jpg")
        #img = Image.open(img_path)
        #quad_im = img.copy()
        #draw = ImageDraw.Draw(img)
        #quad_draw = ImageDraw.Draw(img)
        img = cv2.imread(img_path)
        boxed_img=img
#        offset=5
        ################################

        label = LabelFile(os.path.join(new_labels, file_name + file_ext))
        text_shapes = label.get_all_shapes_for_category('text')
        inhibit_shapes = label.get_all_shapes_for_category('nock')
        arrow_shapes = label.get_all_shapes_for_category('arrow')


        for i,shape in enumerate(label.shapes):
            #a=label.generate_category(shape)
            shape['component']=[]

            if len(shape['points']) == 2:
                shape['points'] = generate_rect_points(shape).tolist()
                shape['shape_type']='polygon'

        for i, shape in enumerate(label.shapes):
            if shape['label'].split(':')[0] =='activate' or shape['label'].split(':')[0] == 'inhibit':
                #shape['componuent']=[0,0]
                rotated_bounding_box=generate_gene_rotated_box(shape['points'])
                shape['rotated_box'] = []
                shape['rotated_box'].append(rotated_bounding_box[0][0])
                shape['rotated_box'].append(rotated_bounding_box[0][1])
                shape['rotated_box'].append(rotated_bounding_box[1][0])
                shape['rotated_box'].append(rotated_bounding_box[1][1])
                shape['rotated_box'].append(rotated_bounding_box[2])



                offset=random.randint(0,5)
                relation = shape['label'].split(':')[1]
                if len(relation.split('|'))>1:
                    #case1: activate:12|5
                    if relation.split('|')[0].isdigit() and relation.split('|')[1].isdigit():

                        start_idx= int(relation.split('|')[0])
                        end_idx = int(relation.split('|')[1])
                        # try:
                        rotated_box ,rect= generate_relation_bounding_box(label.shapes[start_idx]['points'],label.shapes[end_idx]['points'], shape['points'],offset)

                        boxed_img = cv2.drawContours(boxed_img, [rect.astype(int)],-1,(255, 0, 0), 1)
                            # cv2.rectangle(boxed_img, (left_top_x, left_top_y), (right_bottom_x, right_bottom_y), (0, 0, 255),
                            #               1)
                        # except:
                        #     print(file_name,relation)
                        # relation_box=[]
                        # relation_box.append([left_top_x, left_top_y])
                        # relation_box.append([right_bottom_x, right_bottom_y])
                        new_shape = {}
                        new_shape['label'] = 'relation:' +relation
                        new_shape['line_color']=[255,0,0,128]
                        new_shape['fill_color'] = None
                        new_shape['shape_type']='polygon'
                        new_shape['points']=rect.tolist()
                        new_shape['flags']={}
                        try:
                            new_shape['component']=[]
                            new_shape['component'].append(start_idx)
                            new_shape['component'].append(end_idx)
                            new_shape['component'].append(i)
                            new_shape['rotated_box'] = []
                            new_shape['rotated_box'].append(rotated_box[0][0])
                            new_shape['rotated_box'].append(rotated_box[0][1])
                            new_shape['rotated_box'].append(rotated_box[1][0])
                            new_shape['rotated_box'].append(rotated_box[1][1])
                            new_shape['rotated_box'].append(rotated_box[2])
                        except:
                            print("error:"+img_path+str(start_idx)+str(end_idx))

                        label.shapes.append(new_shape)



                    #    continue
                    #case2:activate:[1,2,3]|5
                    elif relation.split('|')[0].find('[')>=0 and relation.split('|')[1].isdigit():
                        end_idx=int(relation.split('|')[1])
                        activate_compound=relation.split('|')[0].replace('[','').replace(']','').split(',')
                        activate_entity_list=[]
                        for i in range(len(activate_compound)):
                            try:
                                activate_entity_list.append(label.shapes[int(activate_compound[i])]['points'])
                            except:
                                print("case2:error:"+img_path+relation)
                        boxed_img, compound_left_top_x, compound_left_top_y, compound_right_bottom_x, compound_right_bottom_y = \
                            generate_compound_bounding_box(boxed_img, activate_entity_list, offset)
                        cv2.rectangle(boxed_img, (compound_left_top_x, compound_left_top_y), (compound_right_bottom_x, compound_right_bottom_y),
                                      (0, 0, 0),2)
                        print("yellow:"+relation)

                        compound=[]
                        compound.append([compound_left_top_x, compound_left_top_y])
                        compound.append([compound_right_bottom_x,compound_right_bottom_y])
                        rotated_box,rect= \
                            generate_relation_bounding_box(compound,label.shapes[end_idx]['points'], shape['points'], offset)
                        boxed_img = cv2.drawContours(boxed_img, [rect.astype(int)],-1,(255, 0, 0), 1)
                        print("red"+relation)
                    # case3:activate:5|[1,2,3]
                    elif relation.split('|')[1].find('[') >= 0 and relation.split('|')[0].isdigit():
                        start_idx = int(relation.split('|')[0])
                        receive_compound = relation.split('|')[1].replace('[', '').replace(']','').split(',')
                        receive_entity_list = []
                        for i in range(len(receive_compound)):
                            try:
                                receive_entity_list.append(label.shapes[int(receive_compound[i])]['points'])
                            except:
                                print("case3:error:"+img_path+relation)
                        boxed_img, compound_left_top_x, compound_left_top_y, compound_right_bottom_x, compound_right_bottom_y = \
                            generate_compound_bounding_box(boxed_img, receive_entity_list, offset)
                        cv2.rectangle(boxed_img, (compound_left_top_x, compound_left_top_y),
                                      (compound_right_bottom_x, compound_right_bottom_y),
                                      (0, 0, 0), 2)
                        print("yellow:" + relation)
                        compound=[]
                        compound.append([compound_left_top_x, compound_left_top_y])
                        compound.append([compound_right_bottom_x,compound_right_bottom_y])
                        rotated_box,rect = \
                            generate_relation_bounding_box(label.shapes[start_idx]['points'],
                                                           compound, shape['points']
                                                           , offset)
                        boxed_img = cv2.drawContours(boxed_img, [rect.astype(int)],-1,(255, 0, 0), 1)
                        print("red"+relation)


                    # case4:activate:[1,2,3]|[4,5]
                    elif relation.split('|')[0].find('[') >= 0 and relation.split('|')[1].find('[') >= 0:
                        activate_compound = relation.split('|')[0].replace('[', '').replace(']','').split(',')
                        activate_entity_list = []
                        receive_compound = relation.split('|')[1].replace('[', '').replace(']','').split(',')
                        receive_entity_list = []
                        for i in range(len(activate_compound)):
                            try:
                                activate_entity_list.append(label.shapes[int(activate_compound[i])]['points'])
                            except:
                                print("case4:error:"+img_path+relation)
                        boxed_img, a_compound_left_top_x, a_compound_left_top_y, a_compound_right_bottom_x, a_compound_right_bottom_y = \
                            generate_compound_bounding_box(boxed_img, activate_entity_list, offset)
                        cv2.rectangle(boxed_img, (a_compound_left_top_x, a_compound_left_top_y), (a_compound_right_bottom_x, a_compound_right_bottom_y),
                                      (0, 0, 0),2)
                        print("yellow:" + relation)
                        a_compound=[]
                        a_compound.append([a_compound_left_top_x, a_compound_left_top_y])
                        a_compound.append([a_compound_right_bottom_x,a_compound_right_bottom_y])


                        for i in range(len(receive_compound)):
                            receive_entity_list.append(label.shapes[int(receive_compound[i])]['points'])
                        boxed_img, r_compound_left_top_x, r_compound_left_top_y, r_compound_right_bottom_x, r_compound_right_bottom_y = \
                            generate_compound_bounding_box(boxed_img, receive_entity_list, offset)
                        cv2.rectangle(boxed_img, (r_compound_left_top_x, r_compound_left_top_y),
                                      (r_compound_right_bottom_x, r_compound_right_bottom_y),
                                      (0, 0, 0), 2)
                        print("yellow:" + relation)
                        r_compound=[]
                        r_compound.append([r_compound_left_top_x, r_compound_left_top_y])
                        r_compound.append([r_compound_right_bottom_x,r_compound_right_bottom_y])

                        rotated_box,rect = \
                            generate_relation_bounding_box(a_compound,r_compound, shape['points']
                                                           , offset)
                        boxed_img = cv2.drawContours(boxed_img, [rect.astype(int)],-1,(255, 0, 0), 1)


                        print("red"+relation)

            elif shape['label'].split(':')[0] =='gene':
                rotated_box=generate_gene_rotated_box(shape['points'])
                shape['rotated_box'] = []
                shape['rotated_box'].append(rotated_box[0][0])
                shape['rotated_box'].append(rotated_box[0][1])
                shape['rotated_box'].append(rotated_box[1][0])
                shape['rotated_box'].append(rotated_box[1][1])
                shape['rotated_box'].append(rotated_box[2])
                # shape['number']=i
        cv2.imwrite(os.path.join(new_label_with_box, file_name+".jpg"), boxed_img)
 #                   elif relation.find('[')>=0:

        label.save(os.path.join(new_label_with_box, file_name+ '.json'),
                   label.shapes,
                   label.imagePath,
                   label.imageHeight, label.imageWidth, label.lineColor, label.fillColor,label.otherData,label.flags)

            #d = shape['label'].split(':')[1]
            # try:
            #     d=shape['label'].split(':')[1]
            # except:
            #     print(file_name)
            #        b=label.shapes[start_idx]['points']
            #        c = label.shapes[end_idx]['points']


             #       print(i,start_idx,end_idx,b,c)
#            generate_sub_image_bounding_two_entities(img, shapes[start_idx]['points'],
#                                                    shapes[end_idx]['points'], offset)


