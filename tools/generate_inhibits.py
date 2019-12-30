import numpy as np
import cv2, os
from tools.label_file import LabelFile
from math import sqrt, atan, degrees
import random


def get_arrow_boundingbox(img, arrow_shapes, text_shapes, ceiling_ratio):
  temp_img = img.copy()
  erase_all_text(temp_img, text_shapes)
  binary_image = cv2.adaptiveThreshold(
    cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY),
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY, 11, 10)

  binary_image_INV = cv2.bitwise_not(binary_image)

  # here is find_contour version
  candidate_contours, hierarchy = cv2.findContours(binary_image_INV,
                                                   cv2.RETR_CCOMP,
                                                   cv2.CHAIN_APPROX_NONE)
  del binary_image, temp_img

  box_contour_idx = []
  head_box_list = []

  max_sim_num = int(len(arrow_shapes) * ceiling_ratio)
  selected_num = 0
  for idx in range(0, len(arrow_shapes)):
        head_box = arrow_shapes[idx]['points']
        # draw the box of arrow
        # cv2.rectangle(origin_img, tuple(head_box[0]), tuple(head_box[1]), (r,g,b), \
        #               thickness=2)
        cnt_idx = match_head_and_arrow_contour(img, head_box,
                                               candidate_contours, hierarchy[0])
        # ,detected_list)
        # make a list for aggregating overlapping arrows
        # box_contour_mapping.append({'arrow' : idx,
        #                             'contour' : cnt_idx})
        if cnt_idx[1] == -1 and cnt_idx[2] == -1:
            selected_num = selected_num + 1
            box_contour_idx.append(cnt_idx[0])
            head_box_list.append(head_box)
        del head_box
        if selected_num > max_sim_num:
            break
        else:
            continue

  # arrow_box = pd.DataFrame(box_contour_mapping)
  # del box_contour_mapping

  # drop duplicates in 'arrow', means corresponding contour with overlapping
  #  objects. we only handle single arrow without overlapping
  # arrow_box.drop_duplicates(subset='arrow',keep= False, inplace= True)
  arrow_box_list = []
  for arrow_idx in box_contour_idx:
    # arrow_contour = arrow_box.iloc[arrow_idx]['contour']
    # calculate its boundingbox
    arrow_box_list.append(cv2.minAreaRect(candidate_contours[arrow_idx]))
  del box_contour_idx, candidate_contours
  return arrow_box_list, head_box_list


def erase_all_text(img, text_shapes):
  for text_shape in text_shapes:
    text_box = text_shape['points']
    text_box = np.array(text_box, np.int32).reshape((-1, 2))

    start_x = int(min(text_box[:, 0]))

    start_y = int(min(text_box[:, 1]))

    end_x = int(max(text_box[:, 0]))

    end_y = int(max(text_box[:, 1]))

    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (255, 255, 255),
                  thickness=-1)


def match_head_and_arrow_contour(img, head_box, contours, hierarchy):  # ,
  # detected_list):
  # arrow_head = Polygon(np.array(head_box, np.int32).reshape((-1,2)))
  # detected_list = list(detected_list)
  interaection_area = []
  for i in range(len(contours)):
    # if i in detected_list:
    #     continue
    interaection_area.append(calculate_interaction_area(img, head_box,
                                                        contours[i]))
  if len(interaection_area) != 0:

    idx = interaection_area.index(max(interaection_area))
    del interaection_area
    return [idx, hierarchy[idx][2], hierarchy[idx][3]]

  else:
    del interaection_area
    # no matching arrow
    return []


def calculate_interaction_area(img, head_box, contour):
  # draw the contour on a white image
  mask = np.zeros(img.shape[:2], dtype=np.uint8)
  cv2.drawContours(mask, [contour], -1, color=1, thickness=1)
  # cv2.fillPoly(mask, contour, 1)
  # calculate the sum of pixels in head box
  head_box = np.array(head_box, np.int32).reshape((-1, 2))

  start_x = int(min(head_box[:, 0]))

  start_y = int(min(head_box[:, 1]))

  end_x = int(max(head_box[:, 0]))

  end_y = int(max(head_box[:, 1]))

  area = np.sum(mask[start_y: end_y, start_x: end_x])
  del mask
  return area


def get_distance_between_two_points(point1, point2):
  return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def find_vertex_closer_to_head(point1, point2, head_box):
  # cv2.rectangle(img, tuple(head_box[0]), tuple(head_box[1]), color=(255, 255,
  #                                                                   0), thickness=2)
  # calculate the center of headbox
  head_box = np.array(head_box, dtype=np.int).reshape((-1, 2))
  head_box = np.mean(head_box, axis=0)
  center_x = head_box[0]
  center_y = head_box[1]

  # cv2.circle(img, (int(center_x), int(center_y)), radius=5, color=(
  #   0, 255, 0), thickness=-1)
  # cv2.imwrite(file_name[:-4]+'head.png', img)
  try:
    distance_1 = get_distance_between_two_points(point1, (center_x, center_y))
    distance_2 = get_distance_between_two_points(point2, (center_x, center_y))
  except:
    pass

  if distance_1 <= distance_2:
    if point2[0] != point1[0]:
      slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
      return point1, slope
    else:
      return point1, None
  elif distance_1 >= distance_2:
    if point1[0] != point2[0]:
      slope = (point1[1] - point2[1]) / (point1[0] - point2[0])
      return point2, slope
    else:
      return point2, None


def draw_inhibit_in_arrow_box(img, file_name, arrow_box, head_box):
  # (rect_center_x, rect_center_y), (rect_width, rect_length), angle = \
  #   arrow_box
  # slope = tan(angle)
  # coefficient = sqrt(rect_length**2 / (1 + slope ** 2))

  inhibit_thickness = random.randint(2, 4)
  r = random.randint(0, 255)
  g = random.randint(0, 255)
  b = random.randint(0, 255)

  rect_vertex = np.int0(cv2.boxPoints(arrow_box))
  # determine which line is longer in arrow box
  length1 = sqrt((rect_vertex[0, 0] - rect_vertex[1, 0]) ** 2 + (
      rect_vertex[0, 1] - rect_vertex[1, 1]) ** 2)
  length2 = sqrt((rect_vertex[0, 0] - rect_vertex[3, 0]) ** 2 + (
      rect_vertex[0, 1] - rect_vertex[3, 1]) ** 2)
  if length1 < length2:
    long_vertex1_x = int((rect_vertex[0, 0] + rect_vertex[1, 0]) / 2)
    long_vertex1_y = int((rect_vertex[0, 1] + rect_vertex[1, 1]) / 2)
    long_vertex2_x = int((rect_vertex[2, 0] + rect_vertex[3, 0]) / 2)
    long_vertex2_y = int((rect_vertex[2, 1] + rect_vertex[3, 1]) / 2)

  else:
    long_vertex1_x = int((rect_vertex[0, 0] + rect_vertex[3, 0]) / 2)
    long_vertex1_y = int((rect_vertex[0, 1] + rect_vertex[3, 1]) / 2)
    long_vertex2_x = int((rect_vertex[2, 0] + rect_vertex[1, 0]) / 2)
    long_vertex2_y = int((rect_vertex[2, 1] + rect_vertex[1, 1]) / 2)

  # determine which line is longer in head_box
  if len(head_box) == 2:
    width = abs(head_box[1][0] - head_box[0][0])
    length = abs(head_box[1][1] - head_box[0][1])
  else:
    # try:
    width = sqrt((head_box[0][0] - head_box[1][0]) ** 2 + (head_box[0][1]
                                                           - head_box[1][
                                                             1]) ** 2)
    length = sqrt((head_box[0][0] - head_box[3][0]) ** 2 + (head_box[0][1]
                                                            - head_box[3][
                                                              1]) ** 2)
  # except:
  #     print('error')
  if width > length:
    head_box_length = width
  else:
    head_box_length = length

  cv2.line(img, (long_vertex1_x, long_vertex1_y),
           (long_vertex2_x, long_vertex2_y),
           (r, g, b), thickness=inhibit_thickness)

  # find out the point closer to head box
  (fixed_x, fixed_y), slope = find_vertex_closer_to_head(
      (long_vertex1_x, long_vertex1_y), (long_vertex2_x,
                                         long_vertex2_y), head_box)
  # draw the found point to confirm
  # cv2.circle(img, (fixed_x, fixed_y), 5, (255,0,0), thickness= -1)

  if slope is not None:
    coefficient_x = (head_box_length * slope) / (2 * sqrt(1 + slope ** 2))
    coefficient_y = head_box_length / (2 * sqrt(1 + slope ** 2))

    cv2.line(img, (int(fixed_x + coefficient_x),
                   int(fixed_y - coefficient_y)),
             (int(fixed_x - coefficient_x),
              int(fixed_y + coefficient_y)),
             (r, g, b), thickness=inhibit_thickness)
    if slope != 0:
      box_width = int(length * 1.5)
      box_hight = int(head_box_length * 1.5)
      angel = atan(- 1 / slope)
      box = cv2.boxPoints(
          ((fixed_x, fixed_y), (box_width, box_hight), degrees(angel)))
      box = np.int0(box)
      #cv2.drawContours(img, [box], 0, (0,0,255), thickness= 2)
      return box
    else:
      box = [[fixed_x - 10, int(fixed_y - 0.5 * head_box_length - 5)],
             [fixed_x - 10, int(fixed_y + 0.5 * head_box_length + 5)],
             [fixed_x + 10, int(fixed_y + 0.5 * head_box_length + 5)],
             [fixed_x + 10, int(fixed_y - 0.5 * head_box_length - 5)]]
      box = np.int0(box)
      #cv2.drawContours(img, [box], 0, (0, 0, 255), thickness=2)
      return box
  else:
    cv2.line(img, (int(fixed_x - 0.5 * head_box_length), fixed_y),
             (int(fixed_x + 0.5 * head_box_length), fixed_y),
             (r, g, b), thickness=inhibit_thickness)
    box = [[int(fixed_x - 0.5 * head_box_length - 5), fixed_y - 10],
           [int(fixed_x + 0.5 * head_box_length + 5), fixed_y - 10],
           [int(fixed_x + 0.5 * head_box_length + 5), fixed_y + 10],
           [int(fixed_x - 0.5 * head_box_length - 5), fixed_y + 10]]
    box = np.int0(box)
    #cv2.drawContours(img, [box], 0, (0, 0, 255), thickness=2)
    return box


def generate_inhibit(img, file_name, arrow_boxes, head_boxes):
  # crop arrow boundingbox and inpaint this rectangle region
  # generate mask for inpaint
  mask = np.zeros(img.shape[:2], np.uint8)
  for box in arrow_boxes:
    rect_pts = cv2.boxPoints(box)
    rect_pts = np.int0(rect_pts)
    # remove arrows in original img
    cv2.drawContours(img, [rect_pts], 0, (255, 255, 255), thickness=-1)
    # set pixel imputation area
    cv2.drawContours(mask, [rect_pts], 0, (255, 255, 255), thickness=-1)
    del rect_pts
  # conduct background imputation on specific area
  simulation_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
  inhibit_box_list = []
  for box_idx in range(0, len(arrow_boxes)):
    try:
        box = draw_inhibit_in_arrow_box(simulation_img, file_name, arrow_boxes[
          box_idx], head_boxes[box_idx])
        inhibit_box_list.append(box)
    except:
        print('%s this image may have invalid box annotations!' % file_name)
        inhibit_box_list.append(None)
  return simulation_img, inhibit_box_list


def simulate_inhibit_per_img(img_fold, json_fold, json_file, sim_json_fold, ceiling_ratio):
    label_file = os.path.join(json_fold, json_file)

    label = LabelFile(label_file)
    img_file = os.path.join(img_fold, label.imagePath)  # single img_file

    # find_all_arrows(img_file, label_file)
    img = cv2.imread(img_file)

    arrow_shapes = label.get_all_shapes_for_category('activate')
    text_shapes = label.get_all_shapes_for_category('gene')
    inhibit_shapes = label.get_all_shapes_for_category('inhibit')
    arrow_boxes, head_boxes = get_arrow_boundingbox(img, arrow_shapes,
                                                  text_shapes, ceiling_ratio)

    # sim = generate_inhibit(img, img_file,arrow_boxes, head_boxes)
    sim, bounding_box_list = generate_inhibit(img, img_file, arrow_boxes,
                                            head_boxes)
    # sim_file, bounding_box = os.path.join(sim_json_fold, image)
    sim_file = os.path.join(sim_json_fold, 'sim_inhibit_' + label.imagePath)

    # replace arrow_shapes with inhibit_shapes
    for arrow_shape in arrow_shapes:
        for idx in range(0, len(head_boxes)):
          # test1 = arrow_shape['points']
          # test2 = head_boxes[idx]
          if arrow_shape['points'] == head_boxes[idx] and \
              bounding_box_list[idx] is not None:
              arrow_shape['points'] = bounding_box_list[idx].tolist()
              arrow_shape['label'] = str(arrow_shape['label']).replace('activate',
                                                                       'inhibit')
              arrow_shape['shape_type'] = 'polygon'
              break

    # shapes=arrow+inhibit+text
    shapes = []
    for arrow_shape in arrow_shapes:
        shapes.append(arrow_shape)
    for inhibit_shape in inhibit_shapes:
        shapes.append(inhibit_shape)
    for text_shape in text_shapes:
        shapes.append(text_shape)

    cv2.imwrite(sim_file, sim)
    # save json

    label.save(os.path.join(os.path.splitext(sim_file)[0] + '.json'), shapes, 'sim_inhibit_' + label.imagePath, None,
             None, None, None, {})
    del sim, shapes
#
if __name__ == "__main__":
   img_fold = r'/home/fei/Desktop/original images/'
   json_fold = r'/home/fei/Desktop/new_labels_with_box/'
   sim_json_fold = r'/home/fei/Desktop/sim_inhibit/'
   for json_file in os.listdir(json_fold):
        if os.path.splitext(json_file)[-1] != '.json':
            continue
        simulate_inhibit_per_img(img_fold,json_fold, json_file,sim_json_fold, 0.8)
#