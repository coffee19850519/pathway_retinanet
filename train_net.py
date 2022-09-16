# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DensePose Training Script.

This script is similar to the training script in detectron2/tools.

It is an example of how a user might use detectron2 for a new project.
"""

import os, csv

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results,DatasetEvaluators
from detectron2.utils.logger import setup_logger
from detectron2.solver.build import build_optimizer


from relation_data_tool_old import register_pathway_dataset, PathwayDatasetMapper, register_Kfold_pathway_dataset, generate_scaled_boxes_width_height_angles
from pathway_evaluation import PathwayEvaluator, RegularEvaluator
# print('abc')
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [PathwayEvaluator(dataset_name, cfg, True, False, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=PathwayDatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):

        return build_detection_train_loader(cfg, mapper=PathwayDatasetMapper(cfg, True))

class RegularTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        evaluators = [RegularEvaluator(dataset_name, cfg, True,False, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # customize reszied parameters
    # cfg['INPUT']['MIN_SIZE_TRAIN'] = (20,)
    # cfg['INPUT']['MAX_SIZE_TRAIN'] = 50
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "densepose" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="pathway_parser")
    return cfg


def main(args):
    cfg = setup(args)
    # import the relation_retinanet as meta_arch, so they will be registered
    from relation_retinanet import RelationRetinaNet

    if args.eval_only:

        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        res = Trainer.test(cfg, model, PathwayEvaluator(cfg.DATASETS.TEST[0], cfg, True, False, cfg.OUTPUT_DIR))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

def evaluate_all_checkpoints(args, checkpoint_folder, output_csv_file):
    cfg = setup(args)
    # import the relation_retinanet as meta_arch, so they will be registered
    from relation_retinanet import RelationRetinaNet
    csv_results=[]
    header = []
    for file in os.listdir(checkpoint_folder):

        file_name, file_ext = os.path.splitext(file)
        if file_ext != ".pth" :
            continue

        model = Trainer.build_model(cfg)
        #logger.info("Model:\n{}".format(model))

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            os.path.join(checkpoint_folder, file), resume=False)


        results=Trainer.test(cfg, model,
                             PathwayEvaluator(cfg.DATASETS.TEST[0], cfg, True, False, cfg.OUTPUT_DIR))
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


if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    category_list = ['relation']
    img_path = r'/home/fei/Desktop/test/images/'
    json_path = r'/home/fei/Desktop/test/jsons/'
    checkpoint_folder = r'/run/media/fei/Entertainment/corrected_imbalanced_setting/'
    output_csv_file = os.path.join(checkpoint_folder, 'all_checkpoint_results.csv')
    register_Kfold_pathway_dataset(json_path, img_path, category_list, K=1)
    #register_pathway_dataset(json_path, img_path, category_list)

    parser = default_argument_parser()
    # parser.add_argument("--task", choices=["train", "eval", "data"], required=True)
    args = parser.parse_args()
    assert not args.eval_only
    args.eval_only = True
    args.config_file = r'./Base-RelationRetinaNet.yaml'

    #args.num_gpus = 2
    #print("Command Line Args:", args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )

    launch(
        evaluate_all_checkpoints,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,checkpoint_folder, output_csv_file),
    )
    # cfg = setup(args)
    # generate_scaled_boxes_width_height_angles(cfg.DATASETS.TRAIN[0], cfg)
