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
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
)
from pathway_evaluation import PathwayEvaluator
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from tools.relation_data_tool import register_pathway_dataset, PathwayDatasetMapper, register_Kfold_pathway_dataset

logger = logging.getLogger("pathway_parser")


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg= cfg,dataset_name= dataset_name, mapper= PathwayDatasetMapper(cfg, False))
        evaluator = PathwayEvaluator(
             dataset_name=dataset_name, cfg= cfg, distributed= False,
             allow_cached= False,
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


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # checkpointer = DetectionCheckpointer(
    #     model, cfg.OUTPUT_DIR,
    #     optimizer=optimizer,
    #     scheduler=scheduler
    # )
    #do not load checkpointer's optimizer and scheduler
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR)
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )

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
    data_loader = build_detection_train_loader(cfg, mapper= PathwayDatasetMapper(cfg, True))

    if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
        epoch_num = (data_loader.dataset.sampler._size // cfg.SOLVER.IMS_PER_BATCH) + 1
    else:
        epoch_num = data_loader.dataset.sampler._size // cfg.SOLVER.IMS_PER_BATCH

    # periodic_checkpointer = PeriodicCheckpointer(
    #     checkpointer,
    #     #cfg.SOLVER.CHECKPOINT_PERIOD,
    #     epoch_num,
    #     max_iter=max_iter
    # )

    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        loss_per_epoch = 0.0
        best_loss = 99999.0
        best_val = 0.0
        better_train = False
        better_val = False
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss for loss in loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
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
                # calculate epoch_loss and push to history cache
                storage.put_scalar("epoch_loss", epoch_loss, smoothing_hint=False)
                for writer in writers:
                    writer.write()
                #do validation
                #do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                #comm.synchronize()

                # only save improved checkpoints on epoch_loss
                if best_loss > losses_reduced:
                    best_loss = losses_reduced
                #     better_train = True
                # if best_val < val_results.values('AP'):
                #     best_val = val_results.values()[0]
                #     better_val = True
                # if better_val and better_train:
                    checkpointer.save("model_{:07d}".format(iteration),  **{"iteration": iteration})

                #reset loss_per_epoch
                loss_per_epoch = 0.0
            else:
                # calculate epoch_loss and push to history cache
                storage.put_scalar("epoch_loss", loss_per_epoch / (iteration % epoch_num), smoothing_hint=False)
                # better_train = False
                # better_val = False

            #periodic_checkpointer.step(iteration)



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
    # import the relation_retinanet as meta_arch, so they will be registered
    from relation_retinanet import RelationRetinaNet

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

    do_train(cfg, model)
    return do_test(cfg, model)


if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    category_list = ['relation']
    img_path = r'/home/coffee/Desktop/data/image_0101/'
    json_path = r'/home/coffee/Desktop/data/json_0101/'

    register_Kfold_pathway_dataset(json_path, img_path, category_list, K=1)
    #register_pathway_dataset(json_path, img_path, category_list)

    parser = default_argument_parser()
    # parser.add_argument("--task", choices=["train", "eval", "data"], required=True)
    args = parser.parse_args()
    assert not args.eval_only
    #args.eval_only = True
    args.config_file = r'./Base-RelationRetinaNet.yaml'
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
