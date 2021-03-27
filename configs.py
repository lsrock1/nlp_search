#!/usr/bin/env python
# COPYRIGHT 2020. Fred Fung. Boston University.
"""
Configs for train/evaluate the baseline retrieval model.
"""
from yacs.config import CfgNode as CN

_C = CN()

_C.EXPR_NAME = "EXPR"
_C.LOG_DIR = "experiments/"
# Workflow type, can be either TRAIN or EVAL.
_C.TYPE = "TRAIN"

# CityFlow-NL related configurations.
_C.DATA = CN()
# Path to MTMC imgs of CityFlow Benchmark.
_C.DATA.CITYFLOW_PATH = "data/"
_C.DATA.JSON_PATH = "data/data/train-tracks.json"
_C.DATA.EVAL_TRACKS_JSON_PATH = "data/data/test-tracks.json"
_C.DATA.EVAL_QUERIES_JSON_PATH = "data/data/test-queries.json"
_C.DATA.DICT_PATH = "data/"
_C.DATA.MIN_COUNT = 5
_C.DATA.NUM_IMG = 3
_C.DATA.MAX_SENTENCE = 15

_C.DATA.LOCAL_CROP_SIZE = (128, 128) 
_C.DATA.GLOBAL_SIZE = (384, 384)


# Model specific configurations.
_C.MODEL = CN()

_C.MODEL.RNN = CN()
# _C.MODEL.RNN.EMBEDDING = 512
_C.MODEL.RNN.HIDDEN = 2048
_C.MODEL.RNN.LAYERS = 6

_C.MODEL.CNN = CN()
_C.MODEL.CNN.ATTN_CHANNEL = 2048

# Training configurations
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 1
_C.TRAIN.EPOCH = 10
_C.TRAIN.STEPS = (5, 8)
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.NUM_WORKERS = 12
_C.TRAIN.PRINT_FREQ = 50
_C.TRAIN.LOSS_CLIP_VALUE = 10.
_C.TRAIN.WARMUP_FACTOR = 0.1
_C.TRAIN.WARMUP_EPOCH = 0
_C.TRAIN.LR = CN()
_C.TRAIN.LR.MOMENTUM = 0.9
_C.TRAIN.LR.WEIGHT_DECAY = 0.1
_C.TRAIN.LR.BASE_LR = 0.00003
_C.TRAIN.ALPHA_COLOR = 0.2
_C.TRAIN.ALPHA_TYPE = 0.2
_C.TRAIN.ALPHA_NL_COLOR = 0.2
_C.TRAIN.ALPHA_NL_TYPE = 0.2


# Evaluation configurations
_C.EVAL = CN()
_C.EVAL.RESTORE_FROM = "experiments/checkpoints/CKPT-E9-S28485.pth"
_C.EVAL.QUERY_JSON_PATH = "data/test-queries.json"
_C.EVAL.BATCH_SIZE = 1
_C.EVAL.NUM_WORKERS = 2
_C.EVAL.CONTINUE = ""


def get_default_config():
    return _C.clone()
