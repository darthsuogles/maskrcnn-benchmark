# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from pathlib import Path
import subprocess
import cv2
import torch
from torchvision import transforms as T

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.data.build import build_dataset
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.config import cfg as root_cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train

from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def gather_args(first_arg, *rest_of_args):
    args = first_arg if isinstance(first_arg, list) else [first_arg]
    for arg in rest_of_args:
        args += arg if isinstance(arg, list) else [arg]
    return args


def vars_exist(first_var, *rest_of_vars):
    vars = gather_args(first_var, *rest_of_vars)
    undef_vars = [var for var in vars if str(var) not in globals()]
    if undef_vars:
        print('undefined variables', undef_vars)
        return False
    return True


def git(first_cmd, *rest_of_cmds):
    cmds = gather_args(first_cmd, *rest_of_cmds)
    return subprocess.check_output(["git"] + cmds).decode('ascii').strip('\n')


def source_import(fpath: Path):
    assert fpath.exists()
    import_file("*", str(fpath))


_IS_REPL_MODE = vars_exist('__file__')
_IS_REBUILD_MODEL = vars_exist('cfg', 'model', 'optimizer')
_REPO_ROOT = Path(git("rev-parse", "--show-toplevel"))

source_import(_REPO_ROOT / "demo" / "model_builder.py")

if _IS_REPL_MODE ^ _IS_REBUILD_MODEL:
    cfg = root_cfg.clone()
    cfg.merge_from_file(
        _REPO_ROOT / "configs" / "e2e_faster_rcnn_R_50_C4_1x.yaml")
    cfg.MODEL.DEVICE = "cpu"

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    is_train = True
    is_distributed = False
    start_iter = 0
    num_gpus = 0


def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    shuffle = True
    num_iters = cfg.SOLVER.MAX_ITER

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file("maskrcnn_benchmark.config.paths_catalog",
                                cfg.PATHS_CATALOG, True)
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(dataset_list, transforms, DatasetCatalog,
                             is_train)

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters,
            start_iter)
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders


if _IS_REPL_MODE:

    arguments = {}
    arguments["iteration"] = 0
    output_dir = cfg.OUTPUT_DIR

    save_to_disk = True
    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler,
                                         output_dir, save_to_disk)
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,
        start_iter=arguments["iteration"],
    )
