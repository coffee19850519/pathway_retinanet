# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
import torch
import datetime
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco import file_lock
from detectron2.structures import pairwise_iou_rotated, RotatedBoxes

from detectron2.structures import BoxMode

from detectron2.utils.logger import create_small_table
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.coco_evaluation import COCOEvaluator

class PathwayEval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
       super().__init__(cocoGt, cocoDt, iouType)

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]
        ious = np.zeros((len(dt), len(gt)))
        for j, g in enumerate(gt):
            for i, d in enumerate(dt):
                # create bounds for ignore regions(double the gt bbox)
                gt_rotated_box = RotatedBoxes(torch.tensor(g['bbox'], dtype= torch.float).view(-1,5))
                dt_rotated_box = RotatedBoxes(torch.tensor(d['bbox'], dtype= torch.float).view(-1,5))
                ious[i, j] = pairwise_iou_rotated(gt_rotated_box, dt_rotated_box)
                del gt_rotated_box,dt_rotated_box
        # if p.iouType == 'segm':
        #     g = [g['segmentation'] for g in gt]
        #     d = [d['segmentation'] for d in dt]
        # elif p.iouType == 'bbox':
        #     g = [g['bbox'] for g in gt]
        #     d = [d['bbox'] for d in dt]
        # else:
        #     raise Exception('unknown iouType for iou computation')
        #
        # # compute iou between each dt and gt region
        # iscrowd = [int(o['iscrowd']) for o in gt]
        # ious = maskUtils.iou(d,g,iscrowd)
        del gt, dt
        return ious


class RegularEvaluator(COCOEvaluator):
    """
    Evaluate object proposal, instance detection/segmentation, keypoint detection
    outputs using COCO's metrics and APIs.
    """

    def __init__(self, dataset_name, cfg, distributed, allow_cached, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            # allow_cached

            convert_to_coco_json(dataset_name, cache_path, allow_cached)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = self.instances_to_coco_json(instances, input["image_id"], input['file_name'])
                # prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def instances_to_coco_json(self, instances, img_id, file_name):
        """
        Dump an "Instances" object to a COCO-format json that's used for evaluation.

        Args:
            instances (Instances):
            img_id (int): the image id

        Returns:
            list[dict]: list of json annotations in COCO format.
        """
        num_instance = len(instances)
        if num_instance == 0:
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        results = []
        for k in range(num_instance):
            result = {
                "image_id": img_id,
                "file_name": file_name,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
            }
            results.append(result)
        return results

class PathwayEvaluator(COCOEvaluator):
    """
    Evaluate object proposal, instance detection/segmentation, keypoint detection
    outputs using COCO's metrics and APIs.
    """

    def __init__(self, dataset_name,  cfg, distributed, allow_cached, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                    so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        #super(PathwayEvaluator, self).__init__(dataset_name, cfg, distributed, output_dir)
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(f"json_file was not found in MetaDataCatalog for '{dataset_name}'")
            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            self.convert_rotated_bbox_prediction_to_coco_json(dataset_name, cache_path, allow_cached)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        print(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def convert_rotated_bbox_prediction_to_coco_json(self, dataset_name, output_file,allow_cached):

        PathManager.mkdirs(os.path.dirname(output_file))
        with file_lock(output_file):
            if PathManager.exists(output_file) and allow_cached:
                self._logger.info(f"Cached annotations in COCO format already exist: {output_file}")
            else:
                self._logger.info(f"Converting dataset annotations in '{dataset_name}' to COCO format ...)")
                coco_dict = self.convert_to_coco_dict(dataset_name)
                with PathManager.open(output_file, "w") as json_file:
                    self._logger.info(f"Caching annotations in COCO format: {output_file}")
                    json.dump(coco_dict, json_file)


    def convert_to_coco_dict(self, dataset_name):
        """
        Convert a dataset in detectron2's standard format into COCO json format

        Generic dataset description can be found here:
        https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

        COCO data format description can be found here:
        http://cocodataset.org/#format-data

        Args:
            dataset_name:
                name of the source dataset
                must be registered in DatastCatalog and in detectron2's standard format
        Returns:
            coco_dict: serializable dict in COCO json format
        """

        dataset_dicts = DatasetCatalog.get(dataset_name)
        categories = [
            {"id": id, "name": name}
            for id, name in enumerate(MetadataCatalog.get(dataset_name).thing_classes)
        ]

        self._logger.info("Converting dataset dicts into COCO format")
        coco_images = []
        coco_annotations = []

        for image_id, image_dict in enumerate(dataset_dicts):
            coco_image = {
                "id": image_dict.get("image_id", image_id),
                "width": image_dict["width"],
                "height": image_dict["height"],
                "file_name": image_dict["file_name"],
            }
            coco_images.append(coco_image)

            anns_per_image = image_dict["annotations"]
            for annotation in anns_per_image:
                # create a new dict with only COCO fields
                coco_annotation = {}

                # COCO requirement: XYWH box format
                bbox = torch.tensor(annotation["bbox"],dtype= torch.float).view(-1,5)
                #bbox_mode = annotation["bbox_mode"]
                #bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYWH_ABS)
                bbox = RotatedBoxes(bbox)


                # COCO requirement: instance area
                # Computing areas using bounding boxes
                #bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                area = bbox.area()[0].item()

                # COCO requirement:
                #   linking annotations to images
                #   "id" field must start with 1
                coco_annotation["id"] = len(coco_annotations) + 1
                coco_annotation["image_id"] = coco_image["id"]
                #coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
                coco_annotation["bbox"] = bbox.tensor.view(-1).tolist()
                coco_annotation["area"] = area
                coco_annotation["category_id"] = annotation["category_id"]
                coco_annotation["iscrowd"] = annotation.get("iscrowd", 0)

                # # Add optional fields
                # if "keypoints" in annotation:
                #     coco_annotation["keypoints"] = keypoints
                #     coco_annotation["num_keypoints"] = num_keypoints
                #
                # if "segmentation" in annotation:
                #     coco_annotation["segmentation"] = annotation["segmentation"]

                coco_annotations.append(coco_annotation)
                del bbox
        self._logger.info(
            "Conversion finished, "
            f"num images: {len(coco_images)}, num annotations: {len(coco_annotations)}"
        )

        info = {
            "date_created": str(datetime.datetime.now()),
            "description": "Automatically generated COCO json file for Detectron2.",
        }
        coco_dict = {
            "info": info,
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
            "licenses": None,
        }
        return coco_dict


    def _tasks_from_config(self, cfg):
        return ("bbox",)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning("no valid predictions generated.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "proposals" in self._predictions[0]:
            self._eval_box_proposals()
        if "instances" in self._predictions[0]:
            self._eval_predictions(set(self._tasks))
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks):
        """
        Evaluate self._predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, self._coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas
                )
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"], input['file_name'])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def read_predictions_with_coco_format_from_json_file(self, file_name):
        results = json.load(open(file_name, 'r'))

        df_results = pd.DataFrame(results)
        #get image number
        for image_idx in df_results['image_id'].drop_duplicates().values:
            prediction = {"image_id": image_idx}
            prediction["instances"] = []
            for result in results:
                if result["image_id"] == image_idx:
                    # then start processing a new image's results
                    prediction["instances"].append({
                        "image_id" : result["image_id"],
                        "category_id" : result["category_id"],
                        "bbox" : result["bbox"],
                        "score" : result["score"]})

            self._predictions.append(prediction)
            del prediction
        del  df_results, results

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {metric: float(coco_eval.stats[idx] * 100) for idx, metric in enumerate(metrics)}
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results



def instances_to_coco_json(instances, img_id,file_name):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    #boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "file_name": file_name,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results



def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)

    pathway_eval = PathwayEval(coco_gt, coco_dt, iou_type)


    # Use the COCO default keypoint OKS sigmas unless overrides are specified
    if kpt_oks_sigmas:
        pathway_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)


    pathway_eval.evaluate()
    pathway_eval.accumulate()
    pathway_eval.summarize()

    return pathway_eval



