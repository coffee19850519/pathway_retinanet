import copy
import math
from typing import List
import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.structures import RotatedBoxes
from detectron2.utils.registry import Registry
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator

ANCHOR_GENERATOR_REGISTRY = Registry("RELATION_ANCHOR_GENERATOR")

@ANCHOR_GENERATOR_REGISTRY.register()
class RelationAnchorGenerator(DefaultAnchorGenerator):
    """
       For a set of image sizes and feature maps, computes a set of anchors based on text and arrow/T-bars.
    """
    def  __init__(self, cfg, input_shape: List[ShapeSpec], instance_pred):
        super().__init__()
        # fmt: off
        sizes = cfg.MODEL.ANCHOR_GENERATOR.SIZES
        aspect_ratios = cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS
        self.strides = [x.stride for x in input_shape]
        # fmt: on
        """
        sizes (list[list[int]]): sizes[i] is the list of anchor sizes to use
            for the i-th feature map. If len(sizes) == 1, then the same list of
            anchor sizes, given by sizes[0], is used for all feature maps. Anchor
            sizes are given in absolute lengths in units of the input image;
            they do not dynamically scale if the input image size changes.
        aspect_ratios (list[list[float]]): aspect_ratios[i] is the list of
            anchor aspect ratios to use for the i-th feature map. If
            len(aspect_ratios) == 1, then the same list of anchor aspect ratios,
            given by aspect_ratios[0], is used for all feature maps.
        strides (list[int]): stride of each input feature.
        """
        self.num_features = len(self.strides)
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)
        self.instance_pred = instance_pred


    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[list[Boxes]]: a list of #image elements. Each is a list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
        """
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)

        anchors_in_image = []
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            boxes = RotatedBoxes(anchors_per_feature_map)
            anchors_in_image.append(boxes)

        anchors = [copy.deepcopy(anchors_in_image) for _ in range(num_images)]
        return anchors

