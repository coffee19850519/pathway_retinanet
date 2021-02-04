import cfg
import cv2, os, copy
import numpy as np
from os import path

# generate entity-sub-images from pathway figures
# output: bounding boxes of subimages, bounding boxes of entities, and bounding boxes of all arrow/nocks
# def get_sub_images():
#    print("\n****Generating Subimages****\n")
#
#    # generate all subimages
#    # dictionaries with the original source imagename as the key
#    all_subimages_boxes = {}
#    all_subimages_entity_boxes = {}
#    all_relationship_shapes = {}
#    for imagename in os.listdir(pathway_image_folder):
#
#        if imagename != ".DS_Store":
#            # generates all of the sub-images for entities detected by the OCR
#            # returns sub image boundaries, entity boundaries, and label shapes of entities
#            subimage_boxes, subimage_entity_boxes, relationship_shapes = generate_relationship_shapes(
#                pathway_image_folder, imagename, ocr_sub_image_folder, 10)
#
#            all_subimages_boxes[imagename] = subimage_boxes
#            all_subimages_entity_boxes[imagename] = subimage_entity_boxes
#            all_relationship_shapes[imagename] = relationship_shapes
#
#    return all_subimages_boxes, all_subimages_entity_boxes, all_relationship_shapes


# takes sub_image_filenames and predicted classes and extracts the relationship type and pairs
# returns entity pairs in list of tuples and list of strings (format: "relationship_type:starter|receptor")
def get_relationship_pairs_on_single_image(file_name, element_instances, relation_instances):
    print("\n****Extracting Relationship Pairs****\n")

    # TODO: extract sub-image according to relation boxes

    

    # loop through all of the sub-images that have a relationship
    #original_filename = None
    # sub_image_relationship_marker = {}
    # shapes_to_write = []
    relationship_tuple_pairs = []
    relationship_descriptions = []
    relationship_bounding_boxes = []


    for sub_image in files_with_relationship:

        # sub_image_boxes = {}
        # sub_image_entity_boxes = {}
        # find what original image they come from
        # get that pathway's sub_image_boxes and sub_image_entity_boxes
        # for original_image_name in all_sub_image_boxes:
        #     if sub_image in all_sub_image_boxes.get(original_image_name).keys():
        #         sub_image_boxes = all_sub_image_boxes.get(original_image_name)
        #         sub_image_entity_boxes = all_sub_image_entity_boxes.get(original_image_name)
        #         original_filename = original_image_name
        #         break

        # if sub_image_boxes is None or sub_image_entity_boxes is None:
        #     continue

        # get the boundaries for the current sub image
        box_coordinates = sub_image_boxes.get(sub_image)
        entity_boxes = sub_image_entity_boxes.get(sub_image)
        candidate_shapes = []

        # get all the arrows and nocks for this subimage's original image
        #sub_image_relationship_shapes = all_relationship_shapes.get(original_filename)

        # if sub_image_relationship_shapes is None:
        #     continue

        # loop through all the arrows and nocks
        # if the bounding box for one of the arrows or nocks is inside of the subimage's boundary
        # then add it as a candidate
        for shape in relationship_shapes:
            try:
                # check top left corner
                # shape['points'][0][0] is top_left_x
                # shape['points'][0][1] is top_left_y
                if shape['points'][0][0] > box_coordinates[0] and shape['points'][0][1] > box_coordinates[1]:

                    # check bottom right corner
                    # shape['points'][2][0] is bottom_right_x
                    # shape['points'][2][1] is bottom_right_y
                    if shape['points'][2][0] < box_coordinates[2] and shape['points'][2][1] < box_coordinates[3]:
                        candidate_shapes.append(shape)
            except:
                continue
        # find best candidate
        # TODO handle index variable better
        index = -1
        counter = 0
        smallest_distance = None
        if len(candidate_shapes) > 1:
            for shape in candidate_shapes:
                temp_distance1 = calculate_distance_between_two_boxes(shape['points'], entity_boxes[0])
                temp_distance2 = calculate_distance_between_two_boxes(shape['points'], entity_boxes[1])
                temp_sum = temp_distance1 + temp_distance2
                if smallest_distance is None or temp_sum < smallest_distance:
                    smallest_distance = temp_sum
                    index = counter
                counter += 1
        elif len(candidate_shapes) == 1:
            index = 0
        else:
            continue

        # determine if best candidate is inhibit or activate
        # add relationship to tuple and descriptions lists
        if index != -1:
            starter, receptor = sub_image.split("_", 1)
            #remove the extention
            receptor, _ = os.path.splitext(receptor)
            #remove the potential suffix
            try:
                receptor, _ = receptor.split("_", 1)
            except:
                # no additional suffix
                pass
            # starter = temp_image_name[0]
            # receptor = temp_image_name[1][:-4]
            temp_shape = candidate_shapes[index]
            temp_relationship_dict = {}

            if "inhibit" in temp_shape['label']:
                temp_relationship = (starter, receptor)
                relationship_tuple_pairs.append(temp_relationship)
                relationship_descriptions.append("inhibit:" + starter + "|" + receptor)

                # save bounding boxes of both entities in relationship and the nock bounding box
                temp_relationship_dict['relationship_bounding_box'] = temp_shape['points']
                temp_relationship_dict['entity1_bounding_box'] = entity_boxes[0]
                temp_relationship_dict['entity2_bounding_box'] = entity_boxes[1]
                relationship_bounding_boxes.append(temp_relationship_dict)

            elif "activate" in temp_shape['label']:
                temp_relationship = (starter, receptor)
                relationship_tuple_pairs.append(temp_relationship)
                relationship_descriptions.append("activate:" + starter + "|" + receptor)

                # save bounding boxes of both entities in relationship and the arrow bounding box
                temp_relationship_dict['relationship_bounding_box'] = temp_shape['points']
                temp_relationship_dict['entity1_bounding_box'] = entity_boxes[0]
                temp_relationship_dict['entity2_bounding_box'] = entity_boxes[1]
                relationship_bounding_boxes.append(temp_relationship_dict)

    print(relationship_tuple_pairs)
    print(relationship_descriptions)

    return relationship_tuple_pairs, relationship_descriptions, relationship_bounding_boxes


# uses ocr to get entities
# from ocr results, generate all subimages from pathway based on detected entities
# returns bounding boxes of all subimages, entity boxes for all subimages, and arrow/nock bounding boxes
def generate_relationship_shapes(image_name, image_ext, current_threshold, predict_box, OCR_results, offset):
    #load current original test image
    image_path = os.path.join(cfg.image_folder, image_name + image_ext)
    image = cv2.imread(image_path)

    # get shapes
    relationship_shapes, gene_shapes = get_shapes_from_prediction_box(predict_box, OCR_results)

    # get original image dimensions
    x_dimension = np.shape(image)[0]
    y_dimension = np.shape(image)[1]
    original_img_dimensions = x_dimension * y_dimension

    sub_image_box_coordinates = {}
    sub_image_entity_boxes = {}

    sub_image_folder = os.path.join(cfg.predict_folder, 'relation_' + image_name + '_' + str(current_threshold))
    if not os.path.exists(sub_image_folder):
        os.mkdir(sub_image_folder)

    # compare all genes es = {}
    for start_idx in range(0, len(gene_shapes)):
        for end_idx in range(start_idx + 1, len(gene_shapes)):
            sub_img, sub_boxes, left_top_y, left_top_x, right_bottom_y, right_bottom_x = \
                generate_sub_image_bounding_two_entities(image, gene_shapes[start_idx]['points'],
                                                         gene_shapes[end_idx]['points'], offset)

            # get sub image dimensions
            x_dimension = np.shape(sub_img)[0]
            y_dimension = np.shape(sub_img)[1]
            sub_img_dimensions = x_dimension * y_dimension

            # threshold for filtering unlikely relationships
            if sub_img_dimensions / original_img_dimensions > .2:
                continue

            if sub_img is not None:
                startor = str(gene_shapes[start_idx]['label']).split(':', 1)[1]
                receptor = str(gene_shapes[end_idx]['label']).split(':', 1)[1]

                # removes bad characters from the filename
                startor = startor.replace('/', '')
                receptor = receptor.replace('/', '')
                startor = startor.replace(':', '')
                receptor = receptor.replace(':', '')
                startor = startor.replace('*', '')
                receptor = receptor.replace('*', '')

                # if not os.path.isdir(cfg.relationship_folder):
                #     os.mkdir(cfg.relationship_folder)
                # if not os.path.isdir(cfg.testing_data_folder):
                #     os.mkdir(cfg.testing_data_folder)
                # if not os.path.isdir(cfg.not_classified_folder):
                #     os.mkdir(cfg.not_classified_folder)


                # check if file exists
                # this happens when a gene's name may appear multiple times on a pathway figure
                file_name = startor + '_' + receptor + '.png'
                if path.exists(sub_image_folder):
                    for x in range(2, 2000):
                        if path.exists(os.path.join(sub_image_folder,
                                                    startor + '_' + receptor + '_' + str(x) + '.png')):
                            continue
                        else:
                            file_name = startor + '_' + receptor + '_' + str(x) + '.png'
                            break

                # create file
                cv2.imwrite(os.path.join(sub_image_folder, file_name), sub_img)
                sub_image_box_coordinates[file_name] = (left_top_x, left_top_y, right_bottom_x, right_bottom_y)
                sub_image_entity_boxes[file_name] = (gene_shapes[start_idx]['points'], gene_shapes[end_idx]['points'])

            else:
                print('exception startor:' + gene_shapes[start_idx][
                    'label'] + 'receptor:' + gene_shapes[end_idx]['label'])

    return sub_image_box_coordinates, sub_image_entity_boxes, relationship_shapes, sub_image_folder


# generates label file shapes from prediction results and ocr corrections
# returns relationship shapes (arrow/nock) and gene shapes (text)
def get_shapes_from_prediction_box(predict_gene_box, corrections):
    # loop through each predicted bounding box
    # x1,y1 is top left corner
    # x2,y2 is bottom left corner
    # x3,y3 is bottom right corner
    # x4,y4 is top right corner
    colorarrow = [255, 0, 0, 128]
    colorinhibit = [0, 255, 0, 128]
    colorgene = [0, 0, 0, 128]
    relationship_shapes = []
    gene_shapes = []
    index = 0
    for pre_each in predict_gene_box:
        tempDict = {}
        points = []
        shape, str_coords = str(pre_each).strip().split('\t')

        if shape == 'arrow':
            tempDict['label'] = 'activate:'
            tempDict['line_color'] = colorarrow
            tempDict['fill_color'] = None
            A = str_coords.replace('[', '')
            A = A.replace(']', '')
            A = A.split(',')
            x1 = int(A[0])
            y1 = int(A[1])
            x2 = int(A[2])
            y2 = int(A[3])
            x3 = int(A[4])
            y3 = int(A[5])
            x4 = int(A[6])
            y4 = int(A[7])
            A1 = (x1, y1)
            A2 = (x2, y2)
            A3 = (x3, y3)
            A4 = (x4, y4)
            points.append(A1)
            points.append(A2)
            points.append(A3)
            points.append(A4)
            tempDict['points'] = points
            del (points)
            tempDict['shape_type'] = 'polygon'
            tempDict['alias'] = 'name'
            relationship_shapes.append(tempDict)
        elif shape == 'text':
            if "****" not in corrections[index]:
                tempDict['label'] = 'gene:' + corrections[index]
                tempDict['line_color'] = colorarrow
                tempDict['fill_color'] = None
                A = str_coords.replace('[', '')
                A = A.replace(']', '')
                A = A.split(',')
                x1 = int(A[0])
                y1 = int(A[1])
                x2 = int(A[2])
                y2 = int(A[3])
                x3 = int(A[4])
                y3 = int(A[5])
                x4 = int(A[6])
                y4 = int(A[7])
                A1 = (x1, y1)
                A2 = (x2, y2)
                A3 = (x3, y3)
                A4 = (x4, y4)
                points.append(A1)
                points.append(A2)
                points.append(A3)
                points.append(A4)
                tempDict['points'] = points
                del (points)
                tempDict['shape_type'] = 'polygon'
                tempDict['alias'] = 'name'
                gene_shapes.append(tempDict)
            index += 1
        elif shape == 'nock':
            tempDict['label'] = 'inhibit:'
            tempDict['line_color'] = colorinhibit
            tempDict['fill_color'] = None
            A = str_coords.replace('[', '')
            A = A.replace(']', '')
            A = A.split(',')
            x1 = int(A[0])
            y1 = int(A[1])
            x2 = int(A[2])
            y2 = int(A[3])
            x3 = int(A[4])
            y3 = int(A[5])
            x4 = int(A[6])
            y4 = int(A[7])
            A1 = (x1, y1)
            A2 = (x2, y2)
            A3 = (x3, y3)
            A4 = (x4, y4)
            points.append(A1)
            points.append(A2)
            points.append(A3)
            points.append(A4)
            tempDict['points'] = points
            del (points)
            tempDict['shape_type'] = 'polygon'
            tempDict['alias'] = 'name'
            relationship_shapes.append(tempDict)

    return relationship_shapes, gene_shapes


# generate sub_image and fill entity bounding boxes
def generate_sub_image_bounding_relation(img, entity1_box, entity2_box,
                                             offset):
    entity1_box = np.array(entity1_box, np.int32).reshape((-1, 2))
    entity2_box = np.array(entity2_box, np.int32).reshape((-1, 2))

    # handle if entity boxes are rectangles and NOT polygons
    if entity1_box.shape[0] == 4:
        entity1_box = cv2.boxPoints(cv2.minAreaRect(entity1_box))
        entity1_box = np.int32(entity1_box)
    if entity2_box.shape[0] == 4:
        entity2_box = cv2.boxPoints(cv2.minAreaRect(entity2_box))
        entity2_box = np.int32(entity2_box)

    # initialize starting dimensions of sub-image
    left_top_x = int(min(min(entity1_box[:, 0]), min(entity2_box[:, 0])))
    left_top_y = int(min(min(entity1_box[:, 1]), min(entity2_box[:, 1])))
    right_bottom_x = int(max(max(entity1_box[:, 0]), max(entity2_box[:, 0])))
    right_bottom_y = int(max(max(entity1_box[:, 1]), max(entity2_box[:, 1])))

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

    # check for bad created dimensions
    if (left_top_y == right_bottom_y) or (left_top_x == right_bottom_x):
        return None
    else:
        # exctract sub-image from original
        sub_img = copy.copy(
            img[left_top_y:right_bottom_y, left_top_x: right_bottom_x])

        # correct coordinates for standalone sub-image
        for entity1_idx in range(entity1_box.shape[0]):
            entity1_box[entity1_idx] = entity1_box[entity1_idx] - [left_top_x, left_top_y]
        for entity2_idx in range(entity2_box.shape[0]):
            entity2_box[entity2_idx] = entity2_box[entity2_idx] - [left_top_x, left_top_y]

        # create filled boundary boxes
        if entity1_box.shape[0] == 2:
            cv2.rectangle(sub_img, tuple(entity1_box[0]), tuple(entity1_box[1]),
                          (0, 0, 255), -1)
        else:
            cv2.drawContours(sub_img, [entity1_box], 0, (0, 0, 255), -1)
        if entity2_box.shape[0] == 2:
            cv2.rectangle(sub_img, tuple(entity2_box[0]), tuple(entity2_box[1]),
                          (0, 0, 255), -1)
        else:
            cv2.drawContours(sub_img, [entity2_box], 0, (0, 0, 255), -1)

        # convert boxes into str
        entity1_box = entity1_box.astype(np.int32).reshape((-1,)).tolist()
        entity2_box = entity2_box.astype(np.int32).reshape((-1,)).tolist()
        str_coords = ','.join(map(str, entity1_box))
        str_coords += '\n'
        str_coords += ','.join(map(str, entity2_box))

        return sub_img, str_coords, left_top_y, left_top_x, right_bottom_y, right_bottom_x


# calculates the Euclidian distance between two bounding boxes
def calculate_distance_between_two_boxes(box1, box2):
    if len(box1) == 4:
        # here box1 is a polygon
        # center point to entity1
        center1_x = (box1[0][0] + box1[1][0] + box1[2][0] + box1[3][0]) / 2
        center1_y = (box1[0][1] + box1[1][1] + box1[2][1] + box1[3][1]) / 2
    else:
        # center point to entity1
        center1_x = (box1[0][0] + box1[1][0]) / 2
        center1_y = (box1[0][1] + box1[1][1]) / 2

    if len(box2) == 4:
        # here box1 is a polygon
        # center point to entity1
        center2_x = (box2[0][0] + box2[1][0] + box2[2][0] + box2[3][0]) / 2
        center2_y = (box2[0][1] + box2[1][1] + box2[2][1] + box2[3][1]) / 2
    else:
        # center point to entity2
        center2_x = (box2[0][0] + box2[1][0]) / 2
        center2_y = (box2[0][1] + box2[1][1]) / 2

    # compute their distance
    return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)


if __name__ == "__main__":

    relation_model = load_relation_predict_model(cfg.sub_img_width_for_relation_predict,
                                                 cfg.sub_img_height_for_relation_predict,
                                                 cfg.num_channels)
    # get sub images

    # all_sub_image_boxes, all_sub_image_entity_boxes, all_relationship_shapes = get_sub_images()
    # predict sub images' relationships
    filenames, predicted_classes = predict_relationships(relation_model,
                                                         r'C:\Users\coffe\Desktop\test\predict\relation_pdf_107_MiR-93_8_34_0.95',
                                                         cfg.sub_img_width_for_relation_predict,
                                                         cfg.sub_img_height_for_relation_predict,
                                                         cfg.num_channels)
    # extract and output relationship pairs
    # get_relationship_pairs(all_sub_image_boxes, all_sub_image_entity_boxes, filenames, predicted_classes,
    #                        all_relationship_shapes)

# end of file