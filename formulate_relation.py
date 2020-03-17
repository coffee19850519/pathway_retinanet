import cfg
import cv2, os
import numpy as np


def assign_roles_to_elements(gene_instances_on_sub_image, relation_head_instance_on_sub_image):
    assert len(gene_instances_on_sub_image) == 2


    element_distance0 = calculate_distance_between_two_boxes(relation_head_instance_on_sub_image['perspective_bbox'], gene_instances_on_sub_image.iloc[0]['perspective_bbox'])
    element_distance1 = calculate_distance_between_two_boxes(relation_head_instance_on_sub_image['perspective_bbox'], gene_instances_on_sub_image.iloc[1]['perspective_bbox'])

    if element_distance0 > element_distance1:
        #return gene_instances_on_sub_image.iloc[0]['ocr'] + '<' + relation_head_instance_on_sub_image['category'] + '>' + gene_instances_on_sub_image.iloc[1]['ocr'], \
        return gene_instances_on_sub_image.iloc[0]['perspective_bbox'], gene_instances_on_sub_image.iloc[1]['perspective_bbox']

    else:
        #return gene_instances_on_sub_image.iloc[1]['ocr'] + '<' + relation_head_instance_on_sub_image['category'] + '>' + gene_instances_on_sub_image.iloc[0]['ocr'], \
        return gene_instances_on_sub_image.iloc[1]['perspective_bbox'], gene_instances_on_sub_image.iloc[0]['perspective_bbox']

def detect_all_contours(img):

    binary_image = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                                         255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,
                                         11, 2)

    binary_image_INV = cv2.bitwise_not(binary_image)

    # here is find_contour version
    contours, hierarchy = cv2.findContours(binary_image_INV, cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours)):

        markedImage = img.copy()

    del markedImage

    del binary_image
    return contours, hierarchy[0], binary_image_INV

def  calculate_intersection_and_area(img, element_bbox, contour):
    # draw the contour on a white image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, color= 1, thickness= -1)

    # calculate the sum of pixels in head box
    start_x = min(element_bbox[:, 0])
    start_y = min(element_bbox[:, 1])
    end_x = max(element_bbox[:, 0])
    end_y = max(element_bbox[:, 1])

    intersection = np.sum(mask[start_y : end_y, start_x: end_x])
    area = np.sum(mask)
    del mask
    return intersection, area

def symbol_area_and_contour(img, symbol_bbox, all_contours, contour_hierarchy):
    intersections = []
    areas = []
    for symbol_idx in range(0, len(all_contours)):
        intersection, area = calculate_intersection_and_area(img, symbol_bbox,
                                                          all_contours[symbol_idx])
        intersections.append(intersection)
        areas.append(area)
    matched_index = intersections.index(max(intersections))
    matched_contour = all_contours[matched_index]
    matched_area = areas[matched_index]
    matched_contour_hierarchy = contour_hierarchy[matched_index]
    del intersections, areas
    return matched_area, matched_contour, matched_contour_hierarchy

def erase_all_text_on_image(img, element_instances):
    text_instances = element_instances.loc[element_instances['category_id'] == 'gene']
    for box in text_instances['perspective_bbox']:
        #start_x, start_y, end_x, end_y = box
        cv2.polylines(img, box, isClosed = True, color= (255,255,255), thickness = -1)

def find_largest_area_symbols(image, gene_instances, relation_head_instances):
    img = image.copy()
    erase_all_text_on_image(img, gene_instances)
    candidate_contours, hierarchy, _ = detect_all_contours(img)
    element_symbol_areas = []
    element_symbol_contours = []

    for relation_symbol_idx in range(0, len(relation_head_instances)):
        element_symbol_area, matched_contour, matched_contour_hierarchy = symbol_area_and_contour(img,
                                relation_head_instances.iloc[relation_symbol_idx]['perspective_bbox'], candidate_contours, hierarchy)
        element_symbol_areas.append(element_symbol_area)
        element_symbol_contours.append(matched_contour)
    max_index = element_symbol_areas.index(max(element_symbol_areas))
    del element_symbol_area,img
    return relation_head_instances.iloc[max_index], element_symbol_contours[max_index]

def find_nearest_point(point, candidates):
  dis = []
  for candidate in candidates:
    distance = dist_center(point, candidate)
    dis.append(distance)
  point_idx = dis.index(min(dis))
  del dis
  return candidates[point_idx], point_idx

def find_vertex_for_detected_relation_symbol_by_distance(img, candidates, head_box):
  aggregate_img = img.copy()
  out_head = []
  in_head = []
  cv2.polylines(aggregate_img, [head_box], True, (0, 255, 0), thickness=2)


  for candidate in candidates:
    candidate = candidate[0]
    if cv2.pointPolygonTest(head_box, tuple(candidate),
                            measureDist=False) != -1:
      in_head.append(candidate)
    else:
      out_head.append(candidate)

  # determine receptor
  if len(in_head) != 0:
      receptor_point = np.mean(in_head, axis=0, dtype=np.int32)
  else:
      receptor_point = np.mean(head_box, axis=0, dtype=np.int32)
  if len(out_head) < 1:
      # activate_point= np.array([0,0],dtype = np.int32);
      #under this circumstance, the line is a dash line
      startor_point = None
      startor_neighbor = None
      #activate_slope = None
      receptor_point = None
      receptor_neighbor = None
      #receptor_slope = None
  elif len(out_head) == 1:
      #connected line is straight line
      startor_point = out_head[0]
      # activate_slope = slope(activate_point, receptor_point)
      # receptor_slope = activate_slope
      startor_neighbor = None
      receptor_neighbor = None

  elif len(out_head) == 2:
      # find first connected key-point for calculating slope
      first_point, first_index = find_nearest_point(receptor_point,out_head)
      #receptor_slope = slope(receptor_point, first_point)
      out_head.pop(first_index)
      startor_point = out_head[0]
      #activate_slope = slope(first_point, activate_point)
      startor_neighbor = first_point
      receptor_neighbor = first_point
  else:
      # find first connected key-point for calculating slope
      first_point, first_index  = find_nearest_point(receptor_point, out_head)
      #receptor_slope = slope(receptor_point, first_point)
      out_head.pop(first_index)
      receptor_neighbor = first_point
      # from receptor point to find the nearest point until end point
      startor_point = first_point.copy()

      while (len(out_head) > 1):
        startor_point, activate_index = find_nearest_point(startor_point,
                                                            out_head)
        out_head.pop(activate_index)
      # the last point is the farest point to receptor point
      first_point = startor_point
      startor_point = out_head[0]
      #activate_slope = slope(first_point, activate_point)
      startor_neighbor = first_point

  #final check the startor & receptor
  box_center_point = np.mean(head_box, axis=0, dtype=np.int32)
  if  startor_point is not None and receptor_point is not None and  \
      startor_neighbor is not None and receptor_neighbor is not None and\
      dist_center(box_center_point, startor_point) < \
      dist_center(box_center_point, receptor_point):
      #incorret order between startor_point & receptor_point
      temp_point = startor_point
      startor_point = receptor_point
      receptor_point = temp_point
      temp_point = startor_neighbor
      startor_neighbor = receptor_neighbor
      receptor_neighbor = temp_point
  del in_head, out_head
  return startor_point,startor_neighbor, receptor_point, receptor_neighbor

# takes sub_image_filenames and predicted classes and extracts the relationship type and pairs
# returns entity pairs in list of tuples and list of strings (format: "relationship_type:starter|receptor")
def get_gene_pairs_on_relation_sub_image (sub_img, element_instances_on_relation,image_name, image_ext, idx):

    #analyze the element distribution first
    gene_instances_on_relation = element_instances_on_relation.loc[element_instances_on_relation['category_name'] == 'gene']
    relation_symbol_instances_on_relation = element_instances_on_relation.loc[element_instances_on_relation['category_name'] != 'gene']

    #pick the mostlikely relation symbols if more than 1 relation symbol
    relation_head_instance, relation_symbol_contour = \
        find_largest_area_symbols(sub_img, gene_instances_on_relation, relation_symbol_instances_on_relation)


    #if more than 2 genes
    if len(gene_instances_on_relation) > 2:
        # TODO alternative strategy 1: cluster then into 2 groups

        # ongoing strategy: from the relation symbol to find the closest 2 genes
        vertex_candidates = cv2.approxPolyDP(relation_symbol_contour, epsilon=5, closed=True)

        startor_point, startor_neighbor, receptor_point, receptor_neighbor = \
        find_vertex_for_detected_relation_symbol_by_distance(sub_img, vertex_candidates, relation_head_instance['perspective_bbox'])

        try:
            startor, receptor, startor_bbox, receptor_bbox = \
                pair_gene(startor_point, startor_neighbor, receptor_point, receptor_neighbor, gene_instances_on_relation)

            cv2.polylines(sub_img, [startor_bbox], isClosed= True,  color=(0, 255, 0), thickness= 2)
            cv2.polylines(sub_img, [receptor_bbox], isClosed= True, color=(0, 0, 255), thickness=2)

            cv2.imwrite(r'/home/fei/Desktop/vis_results_old/paired/' + image_name + '_' + str(idx) + image_ext, sub_img)

            #return startor + '<' + relation_head_instance['category'] + '>' + receptor
        except Exception as e:
            print(str(e))

    # at last leave only 2 genes/groups and 1
    else:
        startor_bbox,receptor_bbox = \
        assign_roles_to_elements(gene_instances_on_sub_image= gene_instances_on_relation, relation_head_instance_on_sub_image= relation_head_instance)
        cv2.polylines(sub_img, [startor_bbox], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(sub_img, [receptor_bbox], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.imwrite(r'/home/fei/Desktop/vis_results_old/paired/' + image_name + '_' + str(idx) + image_ext, sub_img)
        #return pair_info

def perspective_transform_on_element_bbox(element_normalized_bbox, M):
    return cv2.perspectiveTransform(np.array([element_normalized_bbox], np.float32), M).astype(np.int).reshape(-1, 2)

# generate sub_image and fill entity bounding boxes
def generate_sub_image_bounding_relation(img, relation_instance, element_instances_on_sample, offset):

    # image_name, image_ext = os.path.splitext(os.path.basename(img_file_name))

    # element_boxes= []
    # for idx in relation_instance['cover_entity']:
    #     element_boxes.append(entity_instances.iloc[idx]['normalized_bbox'])

    src_pts = relation_instance['normalized_bbox']

    # corrdinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, relation_instance['bbox'][3]-offset],
                        [0, 0],
                        [relation_instance['bbox'][2]-offset, 0],
                        [relation_instance['bbox'][2]-offset,
                         relation_instance['bbox'][3]-offset]], dtype= np.float32)

    # the perspective transformation matrix
    transform = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts)
    # get all element instances on relation region
    element_instances_on_relation = element_instances_on_sample.iloc[relation_instance['covered_elements']].copy()

    # get bbox after perspective transform
    element_instances_on_relation['perspective_bbox'] = element_instances_on_relation['normalized_bbox'].apply(perspective_transform_on_element_bbox, M =transform)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped_img = cv2.warpPerspective(img, transform, (int(relation_instance['bbox'][2]),
                                                      int(relation_instance['bbox'][3])))

    return warped_img, element_instances_on_relation


    # entity_boxes = np.array(covered_entity_instances, np.int32).reshape((-1, 2))
    # #entity2_box = np.array(entity2_box, np.int32).reshape((-1, 2))
    #
    # # handle if entity boxes are rectangles and NOT polygons
    # if entity1_box.shape[0] == 4:
    #     entity1_box = cv2.boxPoints(cv2.minAreaRect(entity1_box))
    #     entity1_box = np.int32(entity1_box)
    # if entity2_box.shape[0] == 4:
    #     entity2_box = cv2.boxPoints(cv2.minAreaRect(entity2_box))
    #     entity2_box = np.int32(entity2_box)
    #
    # # initialize starting dimensions of sub-image
    # left_top_x = int(min(min(entity1_box[:, 0]), min(entity2_box[:, 0])))
    # left_top_y = int(min(min(entity1_box[:, 1]), min(entity2_box[:, 1])))
    # right_bottom_x = int(max(max(entity1_box[:, 0]), max(entity2_box[:, 0])))
    # right_bottom_y = int(max(max(entity1_box[:, 1]), max(entity2_box[:, 1])))
    #
    # # add some padding to sub-image dimensions
    # left_top_x = left_top_x - offset
    # left_top_y = left_top_y - offset
    #
    # # if dimensions with offset are out of range (negative), then set to zero
    # if (left_top_x < 0):
    #     left_top_x = 0
    # if (left_top_y < 0):
    #     left_top_y = 0
    #
    # # if dimensions with offset are out of range (positive), then set to max edge of original image
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
    #     # exctract sub-image from original
    #     sub_img = copy.copy(
    #         img[left_top_y:right_bottom_y, left_top_x: right_bottom_x])
    #
    #     # correct coordinates for standalone sub-image
    #     for entity1_idx in range(entity1_box.shape[0]):
    #         entity1_box[entity1_idx] = entity1_box[entity1_idx] - [left_top_x, left_top_y]
    #     for entity2_idx in range(entity2_box.shape[0]):
    #         entity2_box[entity2_idx] = entity2_box[entity2_idx] - [left_top_x, left_top_y]
    #
    #     # create filled boundary boxes
    #     if entity1_box.shape[0] == 2:
    #         cv2.rectangle(sub_img, tuple(entity1_box[0]), tuple(entity1_box[1]),
    #                       (0, 0, 255), -1)
    #     else:
    #         cv2.drawContours(sub_img, [entity1_box], 0, (0, 0, 255), -1)
    #     if entity2_box.shape[0] == 2:
    #         cv2.rectangle(sub_img, tuple(entity2_box[0]), tuple(entity2_box[1]),
    #                       (0, 0, 255), -1)
    #     else:
    #         cv2.drawContours(sub_img, [entity2_box], 0, (0, 0, 255), -1)
    #
    #     # convert boxes into str
    #     entity1_box = entity1_box.astype(np.int32).reshape((-1,)).tolist()
    #     entity2_box = entity2_box.astype(np.int32).reshape((-1,)).tolist()
    #     str_coords = ','.join(map(str, entity1_box))
    #     str_coords += '\n'
    #     str_coords += ','.join(map(str, entity2_box))
    #
    #     return sub_img, str_coords, left_top_y, left_top_x, right_bottom_y, right_bottom_x


# calculates the Euclidian distance between two bounding boxes
def calculate_distance_between_two_boxes(box1, box2):

    center1_x, center1_y = center_point_in_box(box1)
    center2_x, center2_y = center_point_in_box(box2)
    # compute their distance
    return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

def center_point_in_box(bbox):
    if len(bbox) == 4:
        # here box1 is a polygon
        # center point to entity1
        center_x = (bbox[0][0] + bbox[1][0] + bbox[2][0] + bbox[3][0]) / 2
        center_y = (bbox[0][1] + bbox[1][1] + bbox[2][1] + bbox[3][1]) / 2
    elif len(bbox) == 2:
        # center point to entity1
        center_x = (bbox[0][0] + bbox[1][0]) / 2
        center_y = (bbox[0][1] + bbox[1][1]) / 2
    else:
        raise Exception('invalid bbox dimension')
    return (center_x, center_y)

def dist(vx, vy, x, y, point):
    QP = [point[0] - x, point[1] - y]
    v = [vx, vy]
    h = np.linalg.norm(np.cross(QP, v) / np.linalg.norm(v))
    return h

def dist_center(point1, point2):
    assert len(point1) == len(point2) == 2
    QP = [point1[0] - point2[0], point1[1] - point2[1]]
    return np.sqrt(QP[0] ** 2 + QP[1] ** 2)

def min_vertex_dist(point, box):
    min_dist = 99999999
    for vertex in box:
        dist_pv = dist_center(point, vertex)
        if dist_pv < min_dist:
            min_dist = dist_pv
    return min_dist


def find_best_text(endpoint, text_bboxes, endpoint_neighbor, #img_diagonal, distance_between_endpoints,
                   reverse_endpoint):
    try:
        x = endpoint[0]
        y = endpoint[1]
        vx = endpoint[0] - endpoint_neighbor[0]
        vy = endpoint[1] - endpoint_neighbor[1]
    except:
        pass
    dist_merge = []
    dist_cs = []
    dist_ls = []

    for text_box in text_bboxes:

        dist_c = min_vertex_dist((x, y),  text_box)
        #dist_l = dist(vx, vy, x, y, center_point_in_box(text_box))
        dist_cs.append(dist_c)
        #dist_ls.append(dist_l)
        #dist_merge.append(dist_c + dist_l)

    nearest_index = np.argmin(dist_cs)

    # if dist_center(center_point_in_box(text_bboxes.iloc[nearest_index]), endpoint) < \
    #         dist_center(center_point_in_box(text_bboxes.iloc[nearest_index]), reverse_endpoint):

        #if dist_cs[nearest_index] < 0.1 * img_diagonal and dist_cs[nearest_index] <= 2 * distance_between_endpoints:
    del   dist_merge, dist_cs, dist_ls
    return nearest_index
        # else:
        #     del dist_merge, dist_cs, dist_ls
        #     return None

    # else:
    #     del dist_merge, dist_cs, dist_ls
    #     return None

def pair_gene(startor, startor_neighbor, receptor, receptor_neighbor, text_instances):

    if receptor is None or startor is None:
        raise Exception('startor or receptor is None')

    dist_ar = dist_center(startor, receptor)

    if startor_neighbor is None or \
            dist_center(startor_neighbor, startor) <= 0.1 * dist_ar:
        startor_neighbor = receptor


    if receptor_neighbor is None or \
            dist_center(receptor_neighbor, receptor) <= 0.1 * dist_ar:
        receptor_neighbor = startor

    best_startor_index = \
        find_best_text(startor, text_instances['perspective_bbox'], startor_neighbor, receptor)

    best_receptor_index = \
        find_best_text(receptor, text_instances['perspective_bbox'], receptor_neighbor, startor)

    if best_startor_index is not None and best_receptor_index is not None:
        dist_text = dist_center(center_point_in_box(text_instances.iloc[best_startor_index]['perspective_bbox']),
                                center_point_in_box(text_instances.iloc[best_receptor_index]['perspective_bbox']))

        if best_receptor_index != best_startor_index and dist_text > dist_ar * 0.8:
            return text_instances.iloc[best_startor_index]['ocr'], text_instances.iloc[best_receptor_index]['ocr'],\
                    text_instances.iloc[best_startor_index]['perspective_bbox'], \
                    text_instances.iloc[best_receptor_index]['perspective_bbox']

        else:
            raise Exception('startor and receptor match to a same gene')
    else:
        raise Exception('cannot match startor or receptor')


def plot_connections(result_folder, image_folder, img_file, connect_regions):

  img = cv2.imread(os.path.join(image_folder, img_file))
  layer = img.copy()  # layer = np.zeros((h,w,3),dtype=np.uint8)
  # layer = np.zeros((h,w,3),dtype=np.uint8)
  for c in connect_regions:
    region1 = c[0]
    region2 = c[1]
    rect1center = (
    int((region1[0] + region1[2]) / 2), int((region1[1] + region1[3]) / 2))
    rect2center = (
    int((region2[0] + region2[2]) / 2), int((region2[1] + region2[3]) / 2))
    # cv2.circle(img,rect1center,int((region1[3]-region1[1])*0.5),(255, 0, 0),thickness=2)
    cv2.rectangle(img, (region1[0], region1[1]),
                  (region1[2], region1[3]),
                  (255, 0, 0), thickness=2)

    # drawrect(img,(region1[0], region1[1]),(region1[2], region1[3]),(255, 0, 0),2)
    cv2.rectangle(img, (region2[0], region2[1]),
                  (region2[2], region2[3]),
                  (255, 0, 0), thickness=2)

    cv2.line(layer, rect1center, rect2center, color=(255, 0, 0), thickness=20)

  overlapping = cv2.addWeighted(img, 0.7, layer, 0.3, 0)
  image_name, image_ext = os.path.splitext(img_file)
  cv2.imwrite(os.path.join(result_folder,image_name+'_paring'+ image_ext), overlapping)
  del img, overlapping

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