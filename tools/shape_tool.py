import cv2
import numpy as np
from shapely.geometry import Polygon

def is_smallpolygon_covered_by_largeone(large_polygon, small_polygon):

    # #fill the two polygons
    # cv2.fillPoly()

    large_polygon_points = np.array(large_polygon, np.int).reshape((-1, 2))
    small_polygon_points = np.array(small_polygon, np.int).reshape((-1, 2))
    for point in small_polygon_points:
        if cv2.pointPolygonTest(contour= large_polygon_points,pt= tuple(point), measureDist= False) == -1:
            del large_polygon_points, small_polygon_points
            return False
    del large_polygon_points, small_polygon_points
    return True



def whether_relation_covers_element(element_box_points, relation_box_points, cover_ratio = 0.7):
    element = Polygon(element_box_points)
    relation = Polygon(relation_box_points)
    if relation.intersection(element).area / element.area >= cover_ratio:
        del element, relation
        return True
    else:
        del element, relation
        return False