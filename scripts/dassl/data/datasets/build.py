'''
Author: Guosy_wxy 1579528809@qq.com
Date: 2024-06-28 12:39:25
LastEditors: Guosy_wxy 1579528809@qq.com
LastEditTime: 2024-07-21 07:21:05
FilePath: /Prompt/mvlpt-master/scripts/dassl/data/datasets/build.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from dassl.utils import Registry, check_availability

DATASET_REGISTRY = Registry("DATASET")


def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    #print("avai_datasets", avai_datasets)
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
