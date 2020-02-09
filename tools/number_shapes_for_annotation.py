import cv2, os

import numpy as np
from tools.label_file import LabelFile

def mark_shape_info_on_image(image, shape):
    pts = np.array(shape['points'], np.int)
    #pts = pts.reshape(shape=(4,2))
    if len(pts) == 2:
        cv2.rectangle(image, tuple(pts[0]), tuple(pts[1]), color= (255, 0, 255), thickness= 1)
    else:
        cv2.polylines(image, [pts], isClosed= True, color= (255, 0, 255), thickness= 1)
    cv2.putText(image, str(LabelFile.get_shape_index(shape)), (pts[0,0] + 3, pts[0,1] - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0, 255, 0), 1)


def number_shapes_and_mark_on_image(label_path, image_path, save_image_folder ,save_label_folder):
    try:
        label = LabelFile(label_path)
        label_file_name = os.path.basename(label_path)
        image = cv2.imread(image_path)
        image_file_name = os.path.basename(image_path)
    except:
        print('cannot handle this file: '+ label_path)
        return
    relation_shapes = []
    for shape_idx, shape in enumerate(label.shapes):
    #     #index it
    #     #shape['ID'] = shape_idx
    #
        #plot the boxes and numbers of genes on image
        if LabelFile.get_shape_category(shape) == 'gene':
            mark_shape_info_on_image(image, shape)

        if LabelFile.get_shape_category(shape).find('relation') != -1:
            relation_shapes.append(shape)
        #modify the illegal categories into 'gene'
        # if str(shape['label'].split(':')[0]).find('activate') == -1 and \
        #     str(shape['label'].split(':')[0]).find('inhibit') == -1:
        #     shape['label'] = shape['label'].replace(shape['label'].split(':')[0], 'gene', 1)

    #replace gene names in relation annotations to gene shapes' ID
    # for shape in label.shapes:
    #     shape_category, shape_content = shape['label'].split(':', 1)
    #     if shape_category != 'gene':
    #         #do replacement
    #         try:
    #             gene1, gene2 = shape_content.split('|', 1)
    #         except:
    #             #skip the illegal shape label
    #             continue
    #         gene1_ID = label.get_gene_ID_by_name(gene1)
    #         gene2_ID = label.get_gene_ID_by_name(gene2)
    #         shape['label'] = shape_category + ':' + str(gene1_ID) + '|' + str(gene2_ID)


    #save the number info into json file
    #label.save(os.path.join(save_label_folder, label_file_name), label.shapes, label.imagePath, label.imageHeight, label.imageWidth)

    #save new shape to json file
    label.save(os.path.join(save_label_folder, label_file_name), relation_shapes, label.imagePath, label.imageHeight,
               label.imageWidth)
    #save the marked image
    cv2.imwrite(os.path.join(save_image_folder, image_file_name), image)

    del relation_shapes
    #return duplicated gene info
    #return label.get_duplicated_genes()

def append_modified_relation_annotations(json_with_relation_folder, modified_relation_json_folder, save_new_json_folder):
    for modified_relation_file in os.listdir(modified_relation_json_folder):
        modified_relation_label = LabelFile(os.path.join(modified_relation_json_folder, modified_relation_file))
        json_with_relation_label = LabelFile(os.path.join(json_with_relation_folder, modified_relation_file))
        merged_shapes = []
        #remove all original relation-shapes in json
        for shape in json_with_relation_label.shapes:
            if str(LabelFile.get_shape_category(shape)).find('relation') == -1:
                merged_shapes.append(shape)

        for shape in modified_relation_label.shapes:
            #re-create relations' rotated bbox due to possible adjustment to bbox
            bbox = np.array(shape['points'], np.int32).reshape((-1, 2))
            shape['rotated_box'] = cv2.minAreaRect(bbox)
            #append to original labels
            if LabelFile.get_shape_annotation(shape) != '':
                #only relation shapes without label annotation are invalid
                merged_shapes.append(shape)
        json_with_relation_label.save(os.path.join(save_new_json_folder, modified_relation_file),
                            merged_shapes,
                            json_with_relation_label.imagePath,
                            json_with_relation_label.imageHeight,
                            json_with_relation_label.imageWidth)
        del modified_relation_label, json_with_relation_label,merged_shapes

def pick_images_for_jsons(json_folder,image_folder,save_picked_image_folder):
    for json_file in os.listdir(json_folder):
        json_label = LabelFile(os.path.join(json_folder, json_file))
        img = cv2.imread(os.path.join(image_folder, json_label.imagePath))
        cv2.imwrite(os.path.join(save_picked_image_folder, json_label.imagePath), img)
        del json_label, img


if __name__ == '__main__':
    json_with_relation_folder = r'/home/fei/Desktop/train_data/missing_id_group_work/jsons_with_relations/'
    modified_relation_json_folder = r'/home/fei/Desktop/train_data/missing_id_group_work/relation_jsons/'
    save_new_json_folder = r'/home/fei/Desktop/train_data/missing_id_group_work/merge_jsons/'
    image_folder = r'/home/fei/Desktop/train_data/major/image/'
    save_picked_image_folder = r'/home/fei/Desktop/train_data/missing_id_group_work/images/'
    pick_images_for_jsons(save_new_json_folder, image_folder, save_picked_image_folder)

    #append_modified_relation_annotations(json_with_relation_folder,modified_relation_json_folder,save_new_json_folder)

    # data_fold = r'/home/fei/Desktop/train_data/major/image/'
    # save_image_folder = r'/home/fei/Desktop/train_data/major/element_images/'
    # save_json_folder = r'/home/fei/Desktop/train_data/major/relation_jsons/'

    # for file in os.listdir(data_fold):
    #     file_name, file_ext = os.path.splitext(file)
    #     if file_ext == '.json':
    #         continue
    #     else:
    #         number_shapes_and_mark_on_image(
    #             os.path.join(data_fold,file_name + '.json'),
    #             os.path.join(data_fold,file),
    #             save_image_folder,
    #             save_json_folder)
    #         # with open(os.path.join(save_json_folder, 'gene.txt'), 'a') as gene_info_fp:
    #         #     gene_info_fp.write(file_name + ':\n' + str(duplicated_gene_dict)+ '\n')
    #         # del duplicated_gene_dict


