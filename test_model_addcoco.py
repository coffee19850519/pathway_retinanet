# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detectron2 training script with a plain training loop.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer features, and also
includes fewer abstraction.
"""

import logging
import os,csv,random
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from contextlib import contextmanager

import detectron2.utils.comm as comm
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    build_batch_data_loader
)
from detectron2.data.samplers import InferenceSampler
from torch.utils.data.sampler import Sampler
from detectron2.data.build import DatasetMapper,get_detection_dataset_dicts,DatasetFromList,MapDataset,trivial_batch_collator
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
    COCOEvaluator,
    DatasetEvaluators
    #RotatedCOCOEvaluator,
)
from coco_evaluation_filename import COCOEvaluator_filename
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from relation_data_tool_old import register_pathway_dataset, PathwayDatasetMapper, register_Kfold_pathway_dataset
from  pathway_evaluation import PathwayEvaluator

logger = logging.getLogger("pathway_parser")

def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg= cfg,dataset_name= dataset_name, mapper= PathwayDatasetMapper(cfg, False))
        evaluator = COCOEvaluator_filename(
            dataset_name=dataset_name, cfg = cfg, distributed= False, allow_cached = False,
            output_dir= os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )

        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def do_validation(cfg, data_loader, model,loss_weights,val_evaluator):

    #
    #
    #
    # val_cls_loss = 0.0
    # val_box_reg_loss = 0.0
    # total = 0
    #
    # #with inference_context(model), torch.no_grad():
    # model.eval()
    # with torch.no_grad():
    #     for inputs in data_loader:
    #
    #         # images = model.preprocess_image(inputs)
    #         # features = model.backbone(images.tensor)
    #         # features = [features[f] for f in model.head_in_features]
    #         #
    #         # anchors = model.anchor_generator(features)
    #         # pred_logits, pred_anchor_deltas = model.head(features)
    #         # pred_logits = [permute_to_N_HWA_K(x,model.num_classes) for x in pred_logits]
    #         # pred_anchor_deltas = [permute_to_N_HWA_K(x,4) for x in pred_anchor_deltas]
    #         #
    #         # gt_instances = [x["instances"].to(model.device) for x in inputs]
    #         #
    #         # gt_labels, gt_boxes = model.label_anchors(anchors, gt_instances)
    #         # loss_dict = model.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)
    #         #
    #         loss_dict = model(inputs)
    #
    #         # print(loss_dict)
    #         losses = sum(loss for loss in loss_dict.values())
    #         assert torch.isfinite(losses).all(), loss_dict
    #         val_cls_loss += loss_dict['loss_cls']
    #         val_box_reg_loss += loss_dict['loss_box_reg']
    #         total += 1
    #         del loss_dict, losses
    #
    # # TODO:: AP metric
    # # model.train()
    #
    # val_cls_loss =  val_cls_loss * loss_weights['loss_cls'] / total
    # val_box_reg_loss = val_box_reg_loss * loss_weights['loss_box_reg'] / total
    # val_loss = val_cls_loss + val_box_reg_loss
    # torch.cuda.empty_cache()
    # return val_loss, val_cls_loss, val_box_reg_loss

    eval_results = inference_on_dataset(model, data_loader, DatasetEvaluators([val_evaluator]))
    
    print("*"*20)
    print(eval_results['bbox']['AP-activate'])
    print(eval_results['bbox']['AP-gene'])
    print(eval_results['bbox']['AP-inhibit'])

    torch.cuda.empty_cache()
    return eval_results['bbox']['AP-activate'], eval_results['bbox']['AP-gene'], eval_results['bbox']['AP-inhibit']


class ValidationSampler(Sampler):
    """
    Produce indices for inference.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        shard_size = (self._size - 1) // self._world_size + 1
        self.begin = shard_size * self._rank
        self.end = min(shard_size * (self._rank + 1), self._size)
        #self._local_indices = range(begin, end)


    def __iter__(self):
        yield from iter(torch.randperm(self.end - self.begin).tolist())

    def __len__(self):
        return self.end - self.begin


def build_detection_validation_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_test_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=True,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)

    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = ValidationSampler(len(dataset))

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, cfg.SOLVER.IMS_PER_BATCH, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    print(model)

    # checkpointer = DetectionCheckpointer(
    #     model, cfg.OUTPUT_DIR,
    #     optimizer=optimizer,
    #     scheduler=scheduler
    # )
    #do not load checkpointer's optimizer and scheduler
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR)
    # start_iter = (
    #     checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    # )
    start_iter = 0
    #model.load_state_dict(optimizer)

    max_iter = cfg.SOLVER.MAX_ITER

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    train_data_loader = build_detection_train_loader(cfg, mapper= DatasetMapper(cfg, True))

    # epoch_data_loader = build_detection_test_loader(cfg=cfg, dataset_name= cfg.DATASETS.TRAIN[0],
    #                                           mapper=PathwayDatasetMapper(cfg, True))


    val_data_loader = build_detection_validation_loader(cfg=cfg, dataset_name= cfg.DATASETS.TEST[0],
                                              mapper=DatasetMapper(cfg, True))

    val_evaluator = COCOEvaluator(dataset_name = cfg.DATASETS.TEST[0], output_dir="output/inference",
                                  tasks = None,
                                   distributed = False)



    if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
        epoch_num = (train_data_loader.dataset.sampler._size // cfg.SOLVER.IMS_PER_BATCH) + 1
    else:
        epoch_num = train_data_loader.dataset.sampler._size // cfg.SOLVER.IMS_PER_BATCH

    # periodic_checkpointer = PeriodicCheckpointer(
    #     checkpointer,
    #     #cfg.SOLVER.CHECKPOINT_PERIOD,
    #     epoch_num,
    #     max_iter=max_iter
    # )

    print(epoch_num)

    logger.info("Starting training from iteration {}".format(start_iter))
    loss_weights = {'loss_cls': 1, 'loss_box_reg': 1}
    with EventStorage(start_iter) as storage:
        loss_per_epoch = 0.0
        # best_loss = 99999.0
        best_val_ap = 0.0
        # better_train = False
        # better_val = False
        for data, iteration in zip(train_data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss for loss in loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() * loss_weights[k]  for k, v in comm.reduce_dict(loss_dict).items()}

            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            #prevent gredient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            #if comm.is_main_process():
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            # if (
            #     # cfg.TEST.EVAL_PERIOD > 0
            #     # and
            #         iteration % epoch_num == 0
            #         #iteration % cfg.TEST.EVAL_PERIOD == 0
            #     and iteration != max_iter
            # ):
            #     do_test(cfg, model)
            #     # Compared to "train_net.py", the test results are not dumped to EventStorage
            #     comm.synchronize()

            loss_per_epoch += losses_reduced
            if iteration % epoch_num == 0 or iteration == max_iter:
                #one complete epoch
                epoch_loss = loss_per_epoch / epoch_num
                #do validation
                #epoch_loss, epoch_cls_loss, epoch_box_reg_loss = do_validation(epoch_data_loader, model, loss_weights)
                # val_loss, val_cls_loss, val_box_reg_loss = do_validation(cfg, val_data_loader, model, loss_weights)

                activate_ap, gene_ap, inhibit_ap = do_validation(cfg, val_data_loader, model, loss_weights, val_evaluator)
                total_ap = activate_ap + gene_ap + inhibit_ap
                if best_val_ap < total_ap or iteration == max_iter: #reach a better model
                    best_val_ap = total_ap
                    checkpointer.save("model_best",  **{"iteration": iteration})
                # calculate epoch_loss and push to history cache
                #if comm.is_main_process():
                storage.put_scalar("epoch_loss", epoch_loss, smoothing_hint=False)
                # storage.put_scalar("epoch_cls_loss", epoch_cls_loss, smoothing_hint=False)
                # storage.put_scalar("epoch_box_reg_loss", epoch_box_reg_loss, smoothing_hint=False)
                storage.put_scalar("activate_ap", activate_ap, smoothing_hint=False)
                storage.put_scalar("gene_ap", gene_ap, smoothing_hint=False)
                storage.put_scalar("inhibit_ap", inhibit_ap, smoothing_hint=False)

                for writer in writers:
                    writer.write()

                # only save improved checkpoints on epoch_loss
                # if best_loss > epoch_loss:
                #     best_loss = epoch_loss
                #     better_train = True
                # if best_val_loss > val_loss:
                #     best_val_loss = val_loss
                #     better_val = True
                #if better_val:
                #checkpointer.save("model_{:07d}".format(iteration),  **{"iteration": iteration})
                #comm.synchronize()
                #reset loss_per_epoch
                loss_per_epoch = 0.0
                # better_train = False
                # better_val = False
            del loss_dict,losses,losses_reduced,loss_dict_reduced
            torch.cuda.empty_cache()
            #periodic_checkpointer.step(iteration)

def evaluate_all_checkpoints(args, checkpoint_folder, output_csv_file):
    cfg = setup(args)
    csv_results=[]
    header = []
    for file in os.listdir(checkpoint_folder):

        file_name, file_ext = os.path.splitext(file)
        if file_ext != ".pth" :
            continue

        model = build_model(cfg)
        logger.info("Model:\n{}".format(model))

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            os.path.join(checkpoint_folder, file), resume=False)
        results=do_test(cfg, model)
        results['bbox'].update(checkpoint=file)
        header = results['bbox'].keys()
        csv_results.append(results['bbox'])
        print('main_results:')
        print(results)
        del results
    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = header)
        writer.writeheader()
        writer.writerows(csv_results)
    csvfile.close()
    del csv_results, header

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # customize reszied parameters
    # cfg['INPUT']['MIN_SIZE_TRAIN'] = (20,)
    # cfg['INPUT']['MAX_SIZE_TRAIN'] = 50
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)
    args.eval_only = True
    # import the relation_retinanet as meta_arch, so they will be registered
    # from relation_retinanet import RelationRetinaNet

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    # do_train(cfg, model)
    do_test(cfg, model)


if __name__ == "__main__":
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # category_list = ['gene','activate','inhibit']
    # img_path = r'/home/fei/Desktop/debug_data/image_0109/'
    # json_path = r'/home/fei/Desktop/debug_data/json_0109/'
    # register_Kfold_pathway_dataset(json_path, img_path, category_list, K=1)
    #register_pathway_dataset(json_path, img_path, category_list)

    # TODO:: change these for annotation
    category_list = ['activate', 'gene', 'inhibit']

    img_path = r'C:/Users/Joshua/Documents/Work/Pathway/Datasets/base_val/img/'
    json_path = r'C:/Users/Joshua/Documents/Work/Pathway/Datasets/base_val/json/'
    register_Kfold_pathway_dataset(json_path, img_path, category_list, K=1)

    parser = default_argument_parser()
    # parser.add_argument("--task", choices=["train", "eval", "data"], required=True)
    args = parser.parse_args()
    assert not args.eval_only
    #args.eval_only = True
    #args.num_gpus = 2
    args.config_file = r'Base-RetinaNet.yaml'
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
